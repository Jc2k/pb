const LED_WATTS: f64 = 10.0;

pub fn session_power_summary(tokens: usize, energy_joules: Option<f64>) -> String {
    let token_label = format_number(tokens);
    let Some(energy_joules) = energy_joules.filter(|value| value.is_finite() && *value >= 0.0)
    else {
        return format!("This session used {token_label} tokens.");
    };

    let energy_label = format_energy(energy_joules);
    if energy_joules == 0.0 {
        return format!("This session used {token_label} tokens and an estimated {energy_label}.");
    }
    let seconds = energy_joules / LED_WATTS;
    let (amount, singular, plural) = if seconds < 90.0 {
        (seconds, "second", "seconds")
    } else if seconds < 90.0 * 60.0 {
        (seconds / 60.0, "minute", "minutes")
    } else {
        (seconds / 3_600.0, "hour", "hours")
    };
    let amount_label = format_amount(amount);
    let noun = if (amount - 1.0).abs() < 0.05 {
        singular
    } else {
        plural
    };
    format!(
        "This session used {token_label} tokens and an estimated {energy_label}—the energy a 10 W LED bulb uses in {amount_label} {noun}."
    )
}

fn format_energy(joules: f64) -> String {
    if joules < 1.0 {
        format!("{joules:.2} J")
    } else if joules < 10.0 {
        format!("{joules:.2} J")
    } else if joules < 100.0 {
        format!("{joules:.1} J")
    } else if joules < 1_000.0 {
        format!("{joules:.0} J")
    } else if joules < 3_600_000.0 {
        let watt_hours = joules / 3_600.0;
        format!(
            "{watt_hours:.precision$} Wh",
            precision = if watt_hours < 10.0 {
                2
            } else if watt_hours < 100.0 {
                1
            } else {
                0
            }
        )
    } else {
        let kwh = joules / 3_600_000.0;
        format!(
            "{kwh:.precision$} kWh",
            precision = if kwh < 10.0 {
                3
            } else if kwh < 100.0 {
                2
            } else {
                1
            }
        )
    }
}

fn format_amount(amount: f64) -> String {
    if amount >= 10.0 {
        format!("{amount:.0}")
    } else if amount >= 1.0 {
        let rounded = (amount * 10.0).round() / 10.0;
        if rounded.fract().abs() < f64::EPSILON {
            format!("{rounded:.0}")
        } else {
            format!("{rounded:.1}")
        }
    } else {
        format!("{amount:.2}")
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
    fn uses_a_fixed_explicit_led_comparison() {
        assert_eq!(
            session_power_summary(13_567, Some(216_000.0)),
            "This session used 13,567 tokens and an estimated 60.0 Wh—the energy a 10 W LED bulb uses in 6 hours."
        );
    }

    #[test]
    fn formats_tiny_energy_without_rounding_to_zero() {
        assert_eq!(
            session_power_summary(460_502, Some(37.872)),
            "This session used 460,502 tokens and an estimated 37.9 J—the energy a 10 W LED bulb uses in 3.8 seconds."
        );
    }

    #[test]
    fn falls_back_to_token_only_summary_without_energy() {
        assert_eq!(
            session_power_summary(999, None),
            "This session used 999 tokens."
        );
    }

    #[test]
    fn preserves_a_measured_zero_energy_estimate() {
        assert_eq!(
            session_power_summary(999, Some(0.0)),
            "This session used 999 tokens and an estimated 0.00 J."
        );
    }
}
