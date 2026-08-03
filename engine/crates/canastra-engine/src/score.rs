//! §13 scoring and the §14 match loop.

use crate::action::RuleViolation;
use crate::deal::deal_hand;
use crate::state::{GOING_OUT_BONUS, GameState, Phase, RED_THREE_VALUE, Seat, TARGET_SCORE, Team};

/// §13: one partnership's score for one hand, itemised.
///
/// Broken out rather than reduced to a single number so a UI can lay out the
/// arithmetic the way the score sheet in §17 does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HandScore {
    /// §10: 200 per dirty canastra, 500 per clean one, 1000 for clean aces.
    pub canastra_bonus: i32,
    /// §11.1: 100 to whoever went out.
    pub going_out_bonus: i32,
    /// §12: ±100 per red 3, the sign set by whether a clean canastra exists.
    pub red_three_bonus: i32,
    /// §13.2: every card on the table counts for its face value.
    pub table_cards: i32,
    /// §13.2: every card still held counts against the partnership.
    pub hand_cards: i32,
}

impl HandScore {
    pub fn total(self) -> i32 {
        self.canastra_bonus + self.going_out_bonus + self.red_three_bonus + self.table_cards
            - self.hand_cards
    }
}

/// §13: score one hand for one partnership.
pub fn score_hand(state: &GameState, team: Team) -> HandScore {
    let table = state.table(team);

    let canastra_bonus = table
        .melds
        .iter()
        .map(|meld| meld.canastra().bonus() as i32)
        .sum();

    let going_out_bonus = match state.went_out {
        Some(seat) if seat.team() == team => GOING_OUT_BONUS,
        _ => 0,
    };

    // §12: the same red 3 is worth +100 or −100 depending on nothing but
    // whether the partnership closed a clean canastra. With four red 3s in the
    // deck that is an 800-point swing, which is why a clean canastra is not
    // really optional.
    let per_three = if state.has_clean_canastra(team) {
        RED_THREE_VALUE
    } else {
        -RED_THREE_VALUE
    };
    let red_three_bonus = per_three * table.red_threes.len() as i32;

    // §12: red 3s sit outside every meld and carry no card value, so they are
    // deliberately absent here.
    let table_cards = table
        .melds
        .iter()
        .map(|meld| meld.card_points() as i32)
        .sum();

    let hand_cards = Seat::ALL
        .iter()
        .filter(|seat| seat.team() == team)
        .flat_map(|seat| state.hand(*seat))
        .map(|card| card.points() as i32)
        .sum();

    HandScore {
        canastra_bonus,
        going_out_bonus,
        red_three_bonus,
        table_cards,
        hand_cards,
    }
}

/// §13 and §14: bank the finished hand, then deal the next one or end the match.
///
/// Kept separate from [`crate::apply`] because it is not a player's move — it is
/// what the table does once a hand has stopped.
pub fn settle_hand(state: &GameState) -> Result<GameState, RuleViolation> {
    if state.phase != Phase::HandOver {
        return Err(RuleViolation::HandNotOver);
    }

    let mut scores = state.scores;
    for team in Team::ALL {
        scores[team.index()] += score_hand(state, team).total();
    }

    // §14: reaching 5000 wins; if both partnerships cross in the same hand the
    // higher score takes it. A dead-even finish decides nothing, so the match
    // plays on (CLAUDE.md clarification #2).
    let [first, second] = scores;
    if first.max(second) >= TARGET_SCORE && first != second {
        let mut over = state.clone();
        over.scores = scores;
        over.phase = Phase::MatchOver;
        return Ok(over);
    }

    // §2: the deal passes to the right for the next hand.
    Ok(deal_hand(
        state.seed,
        state.dealer.next(),
        scores,
        state.hand_number + 1,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::Rig;

    /// A seven-card clean canastra: 500 bonus, 55 in cards.
    const CLEAN: &str = "5S 6S 7S 8S 9S TS JS";
    /// The same run soured by a 2: 200 bonus, 65 in cards.
    const DIRTY: &str = "5S 6S 7S 8S 9S TS 2S";

    fn team(index: u8) -> Team {
        Seat::new(index).expect("valid seat").team()
    }

    /// §13 worked example: the clean canastra `5♥ 6♥ 7♥ 8♥ 9♥ 10♥ J♥` is
    /// 500 in bonus plus 55 in cards — 555 points.
    #[test]
    fn the_spec_worked_example_scores_five_hundred_and_fifty_five() {
        let state = Rig::new().meld(1, "5H 6H 7H 8H 9H TH JH").build();
        let score = score_hand(&state, team(1));
        assert_eq!(score.canastra_bonus, 500);
        assert_eq!(score.table_cards, 55);
        assert_eq!(score.total(), 555);
    }

    #[test]
    fn a_dirty_canastra_is_worth_two_hundred_plus_its_cards() {
        let state = Rig::new().meld(1, DIRTY).build();
        let score = score_hand(&state, team(1));
        assert_eq!(score.canastra_bonus, 200);
        assert_eq!(score.table_cards, 65);
    }

    /// §10: a meld short of seven cards earns no bonus but still scores.
    #[test]
    fn a_meld_below_canastra_length_scores_only_its_cards() {
        let state = Rig::new().meld(1, "5H 6H 7H").build();
        let score = score_hand(&state, team(1));
        assert_eq!(score.canastra_bonus, 0);
        assert_eq!(score.table_cards, 15);
    }

    /// §13.2: "Cartas na mão: subtraem", counted across both partners.
    #[test]
    fn cards_left_in_hand_are_subtracted_from_both_partners() {
        let state = Rig::new().hand(1, "AS JOKER").hand(3, "KS").build();
        let score = score_hand(&state, team(1));
        assert_eq!(score.hand_cards, 15 + 50 + 10);
        assert_eq!(score.total(), -75);
    }

    /// §12: black 3s are free to hold — zero points wherever they sit.
    #[test]
    fn black_threes_in_hand_cost_nothing() {
        let state = Rig::new().hand(1, "3S 3C").build();
        assert_eq!(score_hand(&state, team(1)).total(), 0);
    }

    #[test]
    fn going_out_pays_one_hundred_to_that_partnership_only() {
        let state = Rig::new().meld(1, CLEAN).went_out(1).build();
        assert_eq!(score_hand(&state, team(1)).going_out_bonus, 100);
        assert_eq!(score_hand(&state, team(0)).going_out_bonus, 0);
    }

    /// §12: red 3s pay out when the partnership closed a clean canastra.
    #[test]
    fn red_threes_pay_one_hundred_each_alongside_a_clean_canastra() {
        let state = Rig::new().meld(1, CLEAN).red_threes(1, "3H 3D").build();
        let score = score_hand(&state, team(1));
        assert_eq!(score.red_three_bonus, 200);
        assert_eq!(score.total(), 500 + 55 + 200);
    }

    /// §12: without one they cost the same amount instead. A dirty canastra
    /// does not save them — the swing between the two cases is 400 here.
    #[test]
    fn red_threes_cost_one_hundred_each_without_a_clean_canastra() {
        let state = Rig::new().meld(1, DIRTY).red_threes(1, "3H 3D").build();
        let score = score_hand(&state, team(1));
        assert_eq!(score.red_three_bonus, -200);
        assert_eq!(score.total(), 200 + 65 - 200);
    }

    /// §12: a red 3 "não soma valor de carta" — only the ±100.
    #[test]
    fn red_threes_carry_no_card_value_of_their_own() {
        let state = Rig::new().meld(1, CLEAN).red_threes(1, "3H").build();
        assert_eq!(score_hand(&state, team(1)).table_cards, 55);
    }

    /// §10: a clean canastra of aces is worth 1000.
    #[test]
    fn a_clean_ace_canastra_is_worth_a_thousand() {
        let state = Rig::new().meld(1, "AH AD AS AC AH AD AS").build();
        let score = score_hand(&state, team(1));
        assert_eq!(score.canastra_bonus, 1000);
        assert_eq!(score.table_cards, 7 * 15);
    }

    // ---- §14, the match ----

    #[test]
    fn settling_banks_the_score_and_deals_the_next_hand() {
        let state = Rig::new().meld(1, CLEAN).phase(Phase::HandOver).build();
        let next = settle_hand(&state).unwrap();
        assert_eq!(next.scores, [0, 555]);
        assert_eq!(next.hand_number, 2);
        assert_eq!(next.phase, Phase::AwaitingDraw);
        for hand in &next.hands {
            assert_eq!(hand.len(), 15, "the next hand is dealt fresh");
        }
        assert!(next.tables.iter().all(|table| table.melds.is_empty()));
    }

    /// §2: "O carteador passa para a direita a cada mão."
    #[test]
    fn the_deal_passes_on_between_hands() {
        let state = Rig::new().phase(Phase::HandOver).dealer(0).build();
        let next = settle_hand(&state).unwrap();
        assert_eq!(next.dealer, Seat::new(1).unwrap());
        assert_eq!(next.turn, Seat::new(2).unwrap());
    }

    #[test]
    fn settling_a_hand_that_is_still_running_is_refused() {
        let state = Rig::new().phase(Phase::Melding).build();
        assert_eq!(settle_hand(&state), Err(RuleViolation::HandNotOver));
    }

    /// §14: 5000 ends it.
    #[test]
    fn reaching_five_thousand_ends_the_match() {
        let state = Rig::new()
            .scores(0, 4500)
            .meld(1, CLEAN)
            .phase(Phase::HandOver)
            .build();
        let next = settle_hand(&state).unwrap();
        assert_eq!(next.scores, [0, 5055]);
        assert_eq!(next.phase, Phase::MatchOver);
        assert_eq!(next.winner(), Some(team(1)));
    }

    /// §14: "Se as duas duplas passarem de 5.000 na mesma mão, vence quem tiver
    /// mais pontos."
    #[test]
    fn when_both_pass_five_thousand_the_higher_score_wins() {
        let state = Rig::new()
            .scores(5200, 4500)
            .meld(1, CLEAN)
            .phase(Phase::HandOver)
            .build();
        let next = settle_hand(&state).unwrap();
        assert_eq!(next.scores, [5200, 5055]);
        assert_eq!(next.winner(), Some(team(0)));
    }

    /// CLAUDE.md clarification #2: a dead-even finish plays on.
    #[test]
    fn an_exact_tie_above_five_thousand_plays_another_hand() {
        let state = Rig::new().scores(5000, 5000).phase(Phase::HandOver).build();
        let next = settle_hand(&state).unwrap();
        assert_eq!(next.scores, [5000, 5000]);
        assert_eq!(next.phase, Phase::AwaitingDraw);
        assert_eq!(next.hand_number, 2);
        assert_eq!(next.winner(), None);
    }

    #[test]
    fn a_match_below_the_target_simply_continues() {
        let state = Rig::new().scores(100, 200).phase(Phase::HandOver).build();
        let next = settle_hand(&state).unwrap();
        assert_eq!(next.phase, Phase::AwaitingDraw);
        assert_eq!(next.winner(), None);
    }

    /// Each hand is shuffled from the match seed, so a whole match still
    /// replays from one number.
    #[test]
    fn successive_hands_of_a_match_are_dealt_differently_but_reproducibly() {
        let first = crate::deal::new_game(9);
        let over = Rig::new().phase(Phase::HandOver).build();
        assert_eq!(settle_hand(&over), settle_hand(&over), "reproducible");
        assert_ne!(first.stock, settle_hand(&over).unwrap().stock);
    }
}
