import unittest

from ml.cleaner import (
    normalize_text,
    is_administrative_noise,
    is_broad_kind,
    BROAD_KINDS,
)


class TestNormalizeText(unittest.TestCase):
    def test_collapses_whitespace_and_lowercases(self):
        self.assertEqual(normalize_text("  Заявник    Повідомляє  "),
                         "заявник повідомляє")

    def test_strips_none_to_empty(self):
        self.assertEqual(normalize_text(None), "")

    def test_preserves_ukrainian_letters(self):
        self.assertEqual(normalize_text("Теплопостачання"), "теплопостачання")


class TestIsAdministrativeNoise(unittest.TestCase):
    def test_nadano_nomer_prefix_is_noise(self):
        self.assertTrue(
            is_administrative_noise("Надано номер телефону ГЛ Пенсійного фонду (0800503753).")
        )

    def test_nadano_nomery_prefix_is_noise(self):
        self.assertTrue(
            is_administrative_noise("Надано номери телефонів аварійної служби (1580).")
        )

    def test_typo_variant_is_noise(self):
        self.assertTrue(
            is_administrative_noise("Надно номер телефону гл ОДА (0800 500 238).")
        )

    def test_nadani_nomery_is_noise(self):
        self.assertTrue(
            is_administrative_noise("Надані номери телефонів КБУ (050-123-45-67).")
        )

    def test_real_appeal_is_kept(self):
        self.assertFalse(
            is_administrative_noise("Скарга на водія маршрутного таксі №5.")
        )

    def test_deeper_mention_is_not_noise(self):
        self.assertFalse(
            is_administrative_noise(
                "Заявник просить перевірити дії працівників. Надано номер телефону у довідці."
            )
        )

    def test_empty_content_is_not_noise(self):
        self.assertFalse(is_administrative_noise(""))
        self.assertFalse(is_administrative_noise(None))

    def test_nadano_roz_yasnennia_is_kept(self):
        self.assertFalse(
            is_administrative_noise("Надано роз’яснення стосовно термінів розгляду.")
        )


class TestBroadKinds(unittest.TestCase):
    def test_inshe_is_broad(self):
        self.assertTrue(is_broad_kind("Інше"))

    def test_inshi_pytannia_is_broad(self):
        self.assertTrue(is_broad_kind("Інші питання"))

    def test_null_kind_is_broad(self):
        self.assertTrue(is_broad_kind("null"))
        self.assertTrue(is_broad_kind(None))

    def test_concrete_kind_is_not_broad(self):
        self.assertFalse(is_broad_kind("Теплопостачання"))

    def test_broad_kinds_constant(self):
        self.assertEqual(BROAD_KINDS, {"Інше", "Інші питання", "null", None})


if __name__ == "__main__":
    unittest.main()