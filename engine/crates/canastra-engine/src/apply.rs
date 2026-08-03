//! The state machine: apply an [`Action`] to a [`GameState`].

use crate::action::{Action, RuleViolation};
use crate::card::Card;
use crate::deal::resolve_red_threes;
use crate::meld::Meld;
use crate::state::{GameState, Phase, Seat, TurnContext};

/// Check whether a move is legal, without building the state it would produce.
pub fn validate(state: &GameState, seat: Seat, action: &Action) -> Result<(), RuleViolation> {
    apply(state, seat, action).map(|_| ())
}

/// Apply a move and return the resulting state.
///
/// Pure — `state` is never modified, even on failure. Three things follow from
/// that, all of them load-bearing:
///
/// * a caller can abandon a turn that has dead-ended (§6 wants the opening
///   minimum met inside one turn, and that is only knowable at the discard) by
///   going back to the state it held when the turn began;
/// * a searching bot can clone positions without ceremony;
/// * a half-applied action can never leave a game corrupted.
///
/// `seat` is passed in rather than read from `state.turn` so the engine itself
/// rejects out-of-turn moves. A multiplayer server must never have to trust a
/// client's claim about which player it is.
pub fn apply(state: &GameState, seat: Seat, action: &Action) -> Result<GameState, RuleViolation> {
    if matches!(state.phase, Phase::HandOver | Phase::MatchOver) {
        return Err(RuleViolation::HandIsOver);
    }
    if seat != state.turn {
        return Err(RuleViolation::NotYourTurn {
            current: state.turn,
        });
    }

    let mut next = state.clone();
    match action {
        Action::Draw => draw(&mut next, seat)?,
        Action::KeepDrawnCard => keep_drawn(&mut next, seat)?,
        Action::RefuseDrawnCard => refuse_drawn(&mut next, seat)?,
        Action::LayMeld { cards } => lay_meld(&mut next, seat, cards)?,
        Action::AddToMeld { meld, cards } => add_to_meld(&mut next, seat, *meld, cards)?,
        Action::Discard { card } => discard(&mut next, seat, *card)?,
    }
    Ok(next)
}

fn require_phase(state: &GameState, expected: Phase) -> Result<(), RuleViolation> {
    if state.phase == expected {
        Ok(())
    } else {
        Err(RuleViolation::WrongPhase { phase: state.phase })
    }
}

/// §4.1: draw one card from the stock.
fn draw(state: &mut GameState, seat: Seat) -> Result<(), RuleViolation> {
    require_phase(state, Phase::AwaitingDraw)?;

    if state.turn_context.refusal_available {
        // §3: the lead player is shown the card and then decides. A red 3 is
        // never the card on offer — it goes straight to the table and the next
        // card is shown instead, with the refusal still intact
        // (CLAUDE.md clarification #4).
        let offered = draw_past_red_threes(state, seat)?;
        state.turn_context.pending_refusal = Some(offered);
        state.phase = Phase::AwaitingRefusalChoice;
        return Ok(());
    }

    take_into_hand(state, seat)?;
    state.phase = Phase::Melding;
    Ok(())
}

/// Move one card from the stock into `seat`'s hand, tabling any red 3 that turns
/// up and replacing it (§12).
fn take_into_hand(state: &mut GameState, seat: Seat) -> Result<(), RuleViolation> {
    let drawn = state.stock.pop().ok_or(RuleViolation::StockEmpty)?;
    state.hands[seat.index()].push(drawn);
    resolve_red_threes(state, seat);
    Ok(())
}

/// Turn cards up until one is not a red 3, tabling the red 3s on the way past.
fn draw_past_red_threes(state: &mut GameState, seat: Seat) -> Result<Card, RuleViolation> {
    loop {
        let drawn = state.stock.pop().ok_or(RuleViolation::StockEmpty)?;
        if !drawn.is_red_three() {
            return Ok(drawn);
        }
        state.tables[seat.team().index()].red_threes.push(drawn);
    }
}

/// §3: the lead player accepts the card they were shown.
fn keep_drawn(state: &mut GameState, seat: Seat) -> Result<(), RuleViolation> {
    require_phase(state, Phase::AwaitingRefusalChoice)?;
    let offered = take_offered(state);
    state.hands[seat.index()].push(offered);
    state.phase = Phase::Melding;
    Ok(())
}

/// §3: the lead player throws the first card away and takes another. The second
/// card is compulsory — the refusal is good for exactly one card.
fn refuse_drawn(state: &mut GameState, seat: Seat) -> Result<(), RuleViolation> {
    require_phase(state, Phase::AwaitingRefusalChoice)?;
    let refused = take_offered(state);
    state.discard.push(refused);
    take_into_hand(state, seat)?;
    state.phase = Phase::Melding;
    Ok(())
}

fn take_offered(state: &mut GameState) -> Card {
    state.turn_context.refusal_available = false;
    state
        .turn_context
        .pending_refusal
        .take()
        .expect("AwaitingRefusalChoice always holds the card on offer")
}

/// §4.2: put a new meld on the table.
///
/// The meld is validated before any card leaves the hand, so pointing at cards
/// you do not hold and pointing at cards that are not a meld report the reason
/// that actually applies.
fn lay_meld(state: &mut GameState, seat: Seat, cards: &[Card]) -> Result<(), RuleViolation> {
    require_phase(state, Phase::Melding)?;
    let meld = Meld::new(cards).map_err(|reason| RuleViolation::InvalidMeld { reason })?;
    take_all_from_hand(state, seat, cards)?;
    record_laid(state, cards);
    state.tables[seat.team().index()].melds.push(meld);
    Ok(())
}

/// §4.2: add cards to one of your own partnership's melds.
fn add_to_meld(
    state: &mut GameState,
    seat: Seat,
    index: usize,
    cards: &[Card],
) -> Result<(), RuleViolation> {
    require_phase(state, Phase::Melding)?;
    let team = seat.team().index();

    // Build the extended meld off to the side first: a run of lay-offs has to
    // land all together or not at all.
    let mut extended = state.tables[team]
        .melds
        .get(index)
        .cloned()
        .ok_or(RuleViolation::NoSuchMeld { meld: index })?;
    for &card in cards {
        extended
            .add_card(card)
            .map_err(|reason| RuleViolation::InvalidMeld { reason })?;
    }

    take_all_from_hand(state, seat, cards)?;
    record_laid(state, cards);
    state.tables[team].melds[index] = extended;
    Ok(())
}

fn take_all_from_hand(
    state: &mut GameState,
    seat: Seat,
    cards: &[Card],
) -> Result<(), RuleViolation> {
    for &card in cards {
        let position = state.hands[seat.index()]
            .iter()
            .position(|held| *held == card)
            .ok_or(RuleViolation::CardNotInHand { card })?;
        state.hands[seat.index()].remove(position);
    }
    Ok(())
}

/// §6: remember what has gone down this turn, so the opening minimum can be
/// judged when the turn ends.
///
/// Wild cards count their face value toward the minimum — a Joker is 50 and a 2
/// is 20 — which is simply their card value, so no special case is needed. Red
/// 3s never pass through here: they are not melds and never count (§12).
fn record_laid(state: &mut GameState, cards: &[Card]) {
    state.turn_context.laid_value += cards.iter().map(|card| card.points()).sum::<u32>();
    state.turn_context.laid_anything = true;
}

/// §6: a partnership's first melds must clear the bar within a single turn.
///
/// Checked as the turn ends, because that is the first moment the total is
/// known. A player who lays too little simply cannot finish their turn, and the
/// caller backs out to the state the turn started from.
fn commit_opening(state: &mut GameState, seat: Seat) -> Result<(), RuleViolation> {
    let team = seat.team();
    if state.table(team).opened || !state.turn_context.laid_anything {
        return Ok(());
    }
    let required = state.opening_minimum_for(team);
    let laid = state.turn_context.laid_value;
    if laid < required {
        return Err(RuleViolation::OpeningMinimumNotMet { laid, required });
    }
    state.tables[team.index()].opened = true;
    Ok(())
}

/// §4.3: put one card on the pile, which ends the turn.
fn discard(state: &mut GameState, seat: Seat, card: Card) -> Result<(), RuleViolation> {
    require_phase(state, Phase::Melding)?;
    let position = state.hands[seat.index()]
        .iter()
        .position(|held| *held == card)
        .ok_or(RuleViolation::CardNotInHand { card })?;
    commit_opening(state, seat)?;
    state.hands[seat.index()].remove(position);
    state.discard.push(card);
    end_turn(state);
    Ok(())
}

/// §11.2: when the stock is gone the hand stops here. The player who took the
/// last card has just finished their turn, and nobody after them gets to play —
/// not even to take the pile.
fn end_turn(state: &mut GameState) {
    state.turn_context = TurnContext::default();
    if state.stock.is_empty() {
        state.phase = Phase::HandOver;
        return;
    }
    state.turn = state.turn.next();
    state.phase = Phase::AwaitingDraw;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deal::new_game;
    use crate::meld::MeldError;
    use crate::state::{Phase, Seat};
    use crate::testkit::{Rig, card, cards};

    fn seat(index: u8) -> Seat {
        Seat::new(index).expect("valid seat")
    }

    #[test]
    fn only_the_player_to_move_may_act() {
        let state = Rig::new().stock("6D 7D").turn(1).build();
        assert_eq!(
            apply(&state, seat(2), &Action::Draw),
            Err(RuleViolation::NotYourTurn { current: seat(1) })
        );
    }

    #[test]
    fn nobody_may_act_once_the_hand_is_over() {
        let state = Rig::new().phase(Phase::HandOver).turn(1).build();
        assert_eq!(
            apply(&state, seat(1), &Action::Draw),
            Err(RuleViolation::HandIsOver)
        );
    }

    // ---- §4.1, drawing ----

    #[test]
    fn drawing_takes_the_top_of_the_stock_into_hand() {
        let state = Rig::new().stock("6D 7D").hand(1, "KS").turn(1).build();
        let next = apply(&state, seat(1), &Action::Draw).unwrap();
        assert_eq!(next.hand(seat(1)), cards("KS 7D"));
        assert_eq!(next.stock, cards("6D"));
        assert_eq!(next.phase, Phase::Melding);
    }

    #[test]
    fn you_cannot_draw_twice_in_a_turn() {
        let state = Rig::new().stock("6D 7D").turn(1).build();
        let next = apply(&state, seat(1), &Action::Draw).unwrap();
        assert_eq!(
            apply(&next, seat(1), &Action::Draw),
            Err(RuleViolation::WrongPhase {
                phase: Phase::Melding
            })
        );
    }

    #[test]
    fn drawing_from_an_empty_stock_is_impossible() {
        let state = Rig::new().hand(1, "KS").turn(1).build();
        assert_eq!(
            apply(&state, seat(1), &Action::Draw),
            Err(RuleViolation::StockEmpty)
        );
    }

    /// §12: a red 3 goes to the table the moment it is drawn, and the drawer
    /// takes another card in its place.
    #[test]
    fn drawing_a_red_three_tables_it_and_draws_a_replacement() {
        let state = Rig::new().stock("6D 3H").turn(1).build();
        let next = apply(&state, seat(1), &Action::Draw).unwrap();
        assert_eq!(next.hand(seat(1)), cards("6D"));
        assert_eq!(next.table(seat(1).team()).red_threes, cards("3H"));
        assert!(next.stock.is_empty());
    }

    /// §12: a replacement can itself be a red 3, so the swap repeats.
    #[test]
    fn a_chain_of_red_threes_keeps_drawing() {
        let state = Rig::new().stock("6D 3D 3H").turn(1).build();
        let next = apply(&state, seat(1), &Action::Draw).unwrap();
        assert_eq!(next.hand(seat(1)), cards("6D"));
        assert_eq!(next.table(seat(1).team()).red_threes, cards("3H 3D"));
    }

    // ---- §3, the lead player's one-time refusal ----

    #[test]
    fn the_lead_player_sees_their_first_card_before_deciding() {
        let state = new_game(7);
        let lead = state.turn;
        let drawn = apply(&state, lead, &Action::Draw).unwrap();
        assert_eq!(drawn.phase, Phase::AwaitingRefusalChoice);
        assert!(drawn.turn_context.pending_refusal.is_some());
        assert_eq!(drawn.hand(lead).len(), 15, "the card is not in hand yet");
    }

    #[test]
    fn keeping_the_first_card_puts_it_in_hand() {
        let state = new_game(7);
        let lead = state.turn;
        let drawn = apply(&state, lead, &Action::Draw).unwrap();
        let offered = drawn
            .turn_context
            .pending_refusal
            .expect("a card was shown");
        let kept = apply(&drawn, lead, &Action::KeepDrawnCard).unwrap();
        assert_eq!(kept.hand(lead).len(), 16);
        assert!(kept.hand(lead).contains(&offered));
        assert_eq!(kept.phase, Phase::Melding);
    }

    /// §3: "descarta essa carta imediatamente no lixo e compra outra".
    #[test]
    fn refusing_the_first_card_discards_it_and_draws_another() {
        let state = new_game(7);
        let lead = state.turn;
        let drawn = apply(&state, lead, &Action::Draw).unwrap();
        let refused = drawn
            .turn_context
            .pending_refusal
            .expect("a card was shown");
        let next = apply(&drawn, lead, &Action::RefuseDrawnCard).unwrap();
        assert_eq!(next.discard, vec![refused]);
        assert_eq!(next.hand(lead).len(), 16, "the second card is compulsory");
        assert_eq!(next.phase, Phase::Melding);
    }

    /// §3: the refusal is for the first player of the hand, once.
    #[test]
    fn the_second_player_draws_straight_into_their_hand() {
        let state = new_game(7);
        let lead = state.turn;
        let mut game = apply(&state, lead, &Action::Draw).unwrap();
        game = apply(&game, lead, &Action::KeepDrawnCard).unwrap();
        let thrown = *game.hand(lead).first().expect("a full hand");
        game = apply(&game, lead, &Action::Discard { card: thrown }).unwrap();

        assert_eq!(game.turn, lead.next());
        let drawn = apply(&game, game.turn, &Action::Draw).unwrap();
        assert_eq!(drawn.phase, Phase::Melding, "no refusal for anyone else");
    }

    /// §12 with CLAUDE.md clarification #4: a red 3 turning up as the first card
    /// is tabled and replaced without spending the refusal.
    #[test]
    fn a_red_three_does_not_consume_the_lead_players_refusal() {
        let state = Rig::new()
            .stock("6D 3H")
            .turn(1)
            .refusal_available()
            .build();
        let next = apply(&state, seat(1), &Action::Draw).unwrap();
        assert_eq!(next.table(seat(1).team()).red_threes, cards("3H"));
        assert_eq!(next.phase, Phase::AwaitingRefusalChoice);
        assert_eq!(next.turn_context.pending_refusal, Some(card("6D")));
    }

    #[test]
    fn the_refusal_choice_only_exists_after_the_first_draw() {
        let state = Rig::new().stock("6D").turn(1).build();
        assert_eq!(
            apply(&state, seat(1), &Action::KeepDrawnCard),
            Err(RuleViolation::WrongPhase {
                phase: Phase::AwaitingDraw
            })
        );
    }

    // ---- §4.3, discarding ----

    #[test]
    fn discarding_ends_the_turn_and_passes_to_the_right() {
        let state = Rig::new()
            .hand(1, "KS 4H")
            .stock("9C")
            .phase(Phase::Melding)
            .turn(1)
            .build();
        let next = apply(&state, seat(1), &Action::Discard { card: card("KS") }).unwrap();
        assert_eq!(next.hand(seat(1)), cards("4H"));
        assert_eq!(next.discard, cards("KS"));
        assert_eq!(next.turn, seat(2));
        assert_eq!(next.phase, Phase::AwaitingDraw);
    }

    #[test]
    fn the_new_turn_starts_with_a_clean_slate() {
        let state = Rig::new()
            .hand(1, "KS 4H")
            .stock("9C")
            .phase(Phase::Melding)
            .turn(1)
            .refusal_available()
            .build();
        let next = apply(&state, seat(1), &Action::Discard { card: card("KS") }).unwrap();
        assert!(!next.turn_context.refusal_available);
        assert_eq!(next.turn_context.pending_refusal, None);
        assert_eq!(next.turn_context.laid_value, 0);
    }

    #[test]
    fn you_cannot_discard_a_card_you_do_not_hold() {
        let state = Rig::new()
            .hand(1, "KS")
            .phase(Phase::Melding)
            .turn(1)
            .build();
        assert_eq!(
            apply(&state, seat(1), &Action::Discard { card: card("QS") }),
            Err(RuleViolation::CardNotInHand { card: card("QS") })
        );
    }

    #[test]
    fn you_must_draw_before_discarding() {
        let state = Rig::new().hand(1, "KS").stock("9C").turn(1).build();
        assert_eq!(
            apply(&state, seat(1), &Action::Discard { card: card("KS") }),
            Err(RuleViolation::WrongPhase {
                phase: Phase::AwaitingDraw
            })
        );
    }

    /// §11.2: the player who takes the last card finishes their turn normally,
    /// and the hand ends after their discard.
    #[test]
    fn the_hand_ends_when_the_stock_runs_out() {
        let state = Rig::new().hand(1, "KS 4H").stock("9C").turn(1).build();
        let drawn = apply(&state, seat(1), &Action::Draw).unwrap();
        assert!(drawn.stock.is_empty());
        assert_eq!(drawn.phase, Phase::Melding, "they still finish the turn");

        let done = apply(&drawn, seat(1), &Action::Discard { card: card("KS") }).unwrap();
        assert_eq!(done.phase, Phase::HandOver);
    }

    /// CLAUDE.md clarification #3: it counts the same when the last card leaves
    /// as a red 3's replacement rather than as the turn's own draw.
    #[test]
    fn the_hand_ends_when_a_red_three_replacement_empties_the_stock() {
        let state = Rig::new().hand(1, "KS 4H").stock("6D 3H").turn(1).build();
        let drawn = apply(&state, seat(1), &Action::Draw).unwrap();
        assert!(drawn.stock.is_empty());

        let done = apply(&drawn, seat(1), &Action::Discard { card: card("KS") }).unwrap();
        assert_eq!(done.phase, Phase::HandOver);
    }

    // ---- §4.2, laying down and laying off ----

    fn lay(spec: &str) -> Action {
        Action::LayMeld { cards: cards(spec) }
    }

    #[test]
    fn laying_a_meld_moves_cards_from_hand_to_the_table() {
        let state = Rig::new()
            .hand(1, "6H 7H 8H KS")
            .phase(Phase::Melding)
            .opened(1)
            .turn(1)
            .build();
        let next = apply(&state, seat(1), &lay("6H 7H 8H")).unwrap();
        assert_eq!(next.hand(seat(1)), cards("KS"));
        assert_eq!(next.table(seat(1).team()).melds.len(), 1);
        assert_eq!(next.table(seat(1).team()).melds[0].len(), 3);
    }

    #[test]
    fn you_cannot_lay_cards_you_do_not_hold() {
        let state = Rig::new()
            .hand(1, "6H 7H")
            .phase(Phase::Melding)
            .opened(1)
            .turn(1)
            .build();
        assert_eq!(
            apply(&state, seat(1), &lay("6H 7H 8H")),
            Err(RuleViolation::CardNotInHand { card: card("8H") })
        );
    }

    #[test]
    fn an_illegal_combination_is_not_a_meld() {
        let state = Rig::new()
            .hand(1, "6H 7H 9H")
            .phase(Phase::Melding)
            .opened(1)
            .turn(1)
            .build();
        assert_eq!(
            apply(&state, seat(1), &lay("6H 7H 9H")),
            Err(RuleViolation::InvalidMeld {
                reason: MeldError::NotASequence
            })
        );
    }

    #[test]
    fn laying_off_extends_a_meld_already_on_the_table() {
        let state = Rig::new()
            .hand(1, "9H KS")
            .meld(1, "6H 7H 8H")
            .phase(Phase::Melding)
            .turn(1)
            .build();
        let next = apply(
            &state,
            seat(1),
            &Action::AddToMeld {
                meld: 0,
                cards: cards("9H"),
            },
        )
        .unwrap();
        assert_eq!(next.hand(seat(1)), cards("KS"));
        assert_eq!(next.table(seat(1).team()).melds[0].len(), 4);
    }

    /// §4.2: you build on your own partnership's melds, never the opponents'.
    #[test]
    fn you_cannot_lay_off_onto_the_opponents_meld() {
        let state = Rig::new()
            .hand(1, "9H")
            .meld(0, "6H 7H 8H")
            .phase(Phase::Melding)
            .opened(1)
            .turn(1)
            .build();
        assert_eq!(
            apply(
                &state,
                seat(1),
                &Action::AddToMeld {
                    meld: 0,
                    cards: cards("9H")
                }
            ),
            Err(RuleViolation::NoSuchMeld { meld: 0 })
        );
    }

    // ---- §6, the opening minimum ----

    /// §6 worked example: `Coringa-Q♠-K♠-A♠` is 50+10+10+15 = 85, and one meld
    /// is enough to open.
    #[test]
    fn a_single_big_meld_can_open_the_partnership() {
        let state = Rig::new()
            .hand(1, "JOKER QS KS AS 4D")
            .stock("9C 2C")
            .phase(Phase::Melding)
            .turn(1)
            .build();
        let laid = apply(&state, seat(1), &lay("JOKER QS KS AS")).unwrap();
        assert_eq!(laid.turn_context.laid_value, 85);
        assert!(
            !laid.table(seat(1).team()).opened,
            "not until the turn ends"
        );

        let done = apply(&laid, seat(1), &Action::Discard { card: card("4D") }).unwrap();
        assert!(done.table(seat(1).team()).opened);
    }

    /// §6 worked example: `4♥5♥6♥` plus `9♣10♣J♣` is 15 + 30 = 45, which does
    /// not open — and so the turn cannot be ended.
    #[test]
    fn forty_five_points_does_not_open_the_partnership() {
        let state = Rig::new()
            .hand(1, "4H 5H 6H 9C TC JC 2S")
            .stock("9D 2D")
            .phase(Phase::Melding)
            .turn(1)
            .build();
        let laid = apply(&state, seat(1), &lay("4H 5H 6H")).unwrap();
        let laid = apply(&laid, seat(1), &lay("9C TC JC")).unwrap();
        assert_eq!(laid.turn_context.laid_value, 45);

        assert_eq!(
            apply(&laid, seat(1), &Action::Discard { card: card("2S") }),
            Err(RuleViolation::OpeningMinimumNotMet {
                laid: 45,
                required: 75
            })
        );
    }

    /// §6: the minimum counts everything laid in the turn, not one meld at a time.
    #[test]
    fn the_minimum_totals_everything_laid_in_the_turn() {
        let state = Rig::new()
            .hand(1, "4H 5H 6H 9C TC JC QC KC AC 2S")
            .stock("9D 2D")
            .phase(Phase::Melding)
            .turn(1)
            .build();
        let laid = apply(&state, seat(1), &lay("4H 5H 6H")).unwrap();
        let laid = apply(&laid, seat(1), &lay("9C TC JC QC KC AC")).unwrap();
        assert_eq!(laid.turn_context.laid_value, 15 + 65);

        let done = apply(&laid, seat(1), &Action::Discard { card: card("2S") }).unwrap();
        assert!(done.table(seat(1).team()).opened);
    }

    /// §6: past 2500 the bar rises to 120, so the same 85 no longer opens.
    #[test]
    fn a_partnership_past_twenty_five_hundred_needs_one_hundred_and_twenty() {
        let state = Rig::new()
            .hand(1, "JOKER QS KS AS 4D")
            .stock("9C 2C")
            .scores(0, 2500)
            .phase(Phase::Melding)
            .turn(1)
            .build();
        let laid = apply(&state, seat(1), &lay("JOKER QS KS AS")).unwrap();
        assert_eq!(
            apply(&laid, seat(1), &Action::Discard { card: card("4D") }),
            Err(RuleViolation::OpeningMinimumNotMet {
                laid: 85,
                required: 120
            })
        );
    }

    /// §6: "Depois de aberta, a dupla pode baixar livremente."
    #[test]
    fn an_open_partnership_may_lay_anything() {
        let state = Rig::new()
            .hand(1, "4H 5H 6H 2S")
            .stock("9C 2C")
            .opened(1)
            .phase(Phase::Melding)
            .turn(1)
            .build();
        let laid = apply(&state, seat(1), &lay("4H 5H 6H")).unwrap();
        assert!(apply(&laid, seat(1), &Action::Discard { card: card("2S") }).is_ok());
    }

    /// §6: laying nothing at all is always fine, opened or not.
    #[test]
    fn a_turn_without_melding_never_trips_the_minimum() {
        let state = Rig::new()
            .hand(1, "4H 5H")
            .stock("9C 2C")
            .phase(Phase::Melding)
            .turn(1)
            .build();
        let done = apply(&state, seat(1), &Action::Discard { card: card("4H") }).unwrap();
        assert!(!done.table(seat(1).team()).opened);
    }

    /// §12: a red 3 is not a meld and contributes nothing to the minimum.
    #[test]
    fn red_threes_do_not_count_toward_the_opening_minimum() {
        let state = Rig::new()
            .hand(1, "4H 5H 6H 2S")
            .red_threes(1, "3H 3D")
            .stock("9C 2C")
            .phase(Phase::Melding)
            .turn(1)
            .build();
        let laid = apply(&state, seat(1), &lay("4H 5H 6H")).unwrap();
        assert_eq!(laid.turn_context.laid_value, 15);
        assert!(matches!(
            apply(&laid, seat(1), &Action::Discard { card: card("2S") }),
            Err(RuleViolation::OpeningMinimumNotMet { .. })
        ));
    }

    #[test]
    fn validate_agrees_with_apply_without_producing_a_state() {
        let state = Rig::new().stock("6D").turn(1).build();
        assert_eq!(validate(&state, seat(1), &Action::Draw), Ok(()));
        assert_eq!(
            validate(&state, seat(2), &Action::Draw),
            Err(RuleViolation::NotYourTurn { current: seat(1) })
        );
    }
}
