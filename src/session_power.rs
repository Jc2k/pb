#[derive(Debug, Clone, Copy)]
pub struct FunEnergyUnit {
    pub singular: &'static str,
    pub plural: &'static str,
    pub kwh: f64,
    pub min_amount: f64,
}

// Add new deliberately-non-metric session summary units here. Values are rough,
// memorable approximations intended for playful scale rather than measurement.
const FUN_ENERGY_UNITS: &[FunEnergyUnit] = &[
    FunEnergyUnit {
        singular: "cup of tea",
        plural: "cups of tea",
        kwh: 0.03,
        min_amount: 0.1,
    },
    FunEnergyUnit {
        singular: "phone charge",
        plural: "phone charges",
        kwh: 0.015,
        min_amount: 0.1,
    },
    FunEnergyUnit {
        singular: "second of a cozy LED bulb",
        plural: "seconds of a cozy LED bulb",
        kwh: 0.01 / 3600.0,
        min_amount: 1.0,
    },
    FunEnergyUnit {
        singular: "minute of a cozy LED bulb",
        plural: "minutes of a cozy LED bulb",
        kwh: 0.01 / 60.0,
        min_amount: 1.0,
    },
    FunEnergyUnit {
        singular: "hour of a cozy LED bulb",
        plural: "hours of a cozy LED bulb",
        kwh: 0.01,
        min_amount: 0.1,
    },
    FunEnergyUnit {
        singular: "slice of toast",
        plural: "slices of toast",
        kwh: 0.02,
        min_amount: 0.1,
    },
    FunEnergyUnit {
        singular: "bag of microwave popcorn",
        plural: "bags of microwave popcorn",
        kwh: 0.05,
        min_amount: 0.1,
    },
];

pub fn session_power_summary(tokens: usize, energy_kwh: f64) -> String {
    let token_label = format_number(tokens);
    if energy_kwh <= 0.0 || !energy_kwh.is_finite() {
        return format!("This session used {token_label} tokens.");
    }

    let unit = choose_fun_unit(energy_kwh);
    let amount = energy_kwh / unit.kwh;
    let amount_label = format_amount(amount);
    let noun = if amount_label == "1" {
        unit.singular
    } else {
        unit.plural
    };
    format!(
        "This session used {token_label} tokens and enough electricity for {amount_label} {noun}."
    )
}

fn choose_fun_unit(energy_kwh: f64) -> FunEnergyUnit {
    FUN_ENERGY_UNITS
        .iter()
        .copied()
        .min_by(|left, right| {
            let left_score = unit_score(energy_kwh, *left);
            let right_score = unit_score(energy_kwh, *right);
            left_score.total_cmp(&right_score)
        })
        .expect("fun energy units should not be empty")
}

fn unit_score(energy_kwh: f64, unit: FunEnergyUnit) -> f64 {
    let amount = energy_kwh / unit.kwh;
    if amount < unit.min_amount {
        return f64::INFINITY;
    }
    let nearest_fun_amount = if amount < 1.0 { 1.0 } else { amount.round() };
    (amount - nearest_fun_amount).abs()
}

fn format_amount(amount: f64) -> String {
    if amount >= 10.0 {
        format!("{:.0}", amount)
    } else if amount >= 1.0 {
        let rounded = (amount * 10.0).round() / 10.0;
        if (rounded.fract()).abs() < f64::EPSILON {
            format!("{rounded:.0}")
        } else {
            format!("{rounded:.1}")
        }
    } else {
        format!("{amount:.1}")
    }
}

fn format_number(value: usize) -> String {
    let digits = value.to_string();
    let mut formatted = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, ch) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            formatted.push(',');
        }
        formatted.push(ch);
    }
    formatted
}

#[cfg(test)]
mod tests {
    use super::session_power_summary;

    #[test]
    fn formats_tokens_and_fun_energy_unit() {
        assert_eq!(
            session_power_summary(13_567, 0.06),
            "This session used 13,567 tokens and enough electricity for 2 cups of tea."
        );
    }

    #[test]
    fn formats_tiny_energy_as_nonzero_fun_unit() {
        assert_eq!(
            session_power_summary(460_502, 0.00001052),
            "This session used 460,502 tokens and enough electricity for 3.8 seconds of a cozy LED bulb."
        );
    }

    #[test]
    fn falls_back_to_token_only_summary_without_energy() {
        assert_eq!(
            session_power_summary(999, 0.0),
            "This session used 999 tokens."
        );
    }
}
