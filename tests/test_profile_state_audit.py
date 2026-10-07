import unittest
from copy import deepcopy
from unittest.mock import patch
from scripts.exp4_profile.followup import reader_followup
from atomic_memory.profile_state_audit import audit, reverse_presentation, fact_multiset


class QueryAuditTest(unittest.TestCase):
    def memory(self,text):
        return [dict(memory_id='m00',text=text)]

    def test_cross_subject_value_is_not_support(self):
        m=self.memory("Emma's current job is nurse. Sofia's current city is Seattle.")
        a=audit(m,'Emma',['city','job'],dict(city='Seattle',job='nurse'))
        self.assertFalse(a['queried_profile_complete'])
        self.assertIsNone(a['queried_profile_unique_and_noncontradictory'])
        self.assertFalse(a['answer_supported_by_queried_subject_memory'])

    def test_supported_answer_can_be_contradictory(self):
        m=self.memory("Emma's current city is Seattle. Emma's current city is Boston.")
        a=audit(m,'Emma',['city'],dict(city='Seattle'))
        self.assertTrue(a['queried_profile_complete'])
        self.assertFalse(a['queried_profile_unique_and_noncontradictory'])
        self.assertTrue(a['answer_supported_by_queried_subject_memory'])

    def test_duplicate_is_not_contradiction_and_history_not_current(self):
        m=self.memory("Emma's current city is Seattle. Emma's current city is Seattle. Emma's previous city was Boston.")
        a=audit(m,'Emma',['city'],dict(city='Boston'))
        self.assertTrue(a['queried_profile_unique_and_noncontradictory'])
        self.assertFalse(a['answer_supported_by_queried_subject_memory'])

    def test_partial_parse_does_not_silently_discard_unknown_assertion(self):
        m=self.memory("Emma's current city is Seattle. Emma no longer lives in Seattle.")
        for relaxed in [False,True]:
            a=audit(m,'Emma',['city'],dict(city='Seattle'),relaxed)
            self.assertIsNone(a['queried_profile_complete'])

    def test_relaxation_is_separate_and_closed(self):
        for text in ["Emma's city is Seattle.","Emma's city is now Seattle", "Emma's current city is Seattle"]:
            m=self.memory(text)
            self.assertIsNone(audit(m,'Emma',['city'],dict(city='Seattle'))['queried_profile_complete'])
            self.assertTrue(audit(m,'Emma',['city'],dict(city='Seattle'),True)['answer_supported_by_queried_subject_memory'])
        for text in ["Emma's city is not Seattle.","Emma's city is Seattle sometimes.","Emma's city is Atlantis."]:
            self.assertIsNone(audit(self.memory(text),'Emma',['city'],dict(city='Seattle'),True)['queried_profile_complete'])

    def test_abstention_is_not_an_asserted_supported_value(self):
        a=audit(self.memory("Emma's current city is Seattle."),'Emma',['city'],dict(city='UNKNOWN'))
        self.assertIsNone(a['answer_supported_by_queried_subject_memory'])

    def test_order_reversal_preserves_facts_memberships_and_is_involution(self):
        m=[dict(memory_id='m00',text="Emma's current city is Seattle. Emma's current city is Boston."),
           dict(memory_id='m01',text="Sofia's current job is nurse.")]
        v=reverse_presentation(m)
        self.assertEqual(fact_multiset(m),fact_multiset(v))
        self.assertEqual(m,reverse_presentation(v))
        self.assertTrue(v[1]['text'].startswith("Emma's current city is Boston."))


class FollowupFreezeTest(unittest.TestCase):
    def test_offline_data_check_and_generation_source_guard(self):
        manifest = reader_followup.verify(check_code=False)
        self.assertEqual(len(manifest['source_sha256']), 8)
        matching = deepcopy(manifest)
        matching['code_sha256'] = {
            name: reader_followup.digest(reader_followup.ROOT / name)
            for name in manifest['code_sha256']
        }
        with patch.object(reader_followup, 'load', return_value=matching):
            self.assertEqual(reader_followup.verify(), matching)
        mismatched = deepcopy(matching)
        name = next(iter(mismatched['code_sha256']))
        mismatched['code_sha256'][name] = '0' * 64
        with patch.object(reader_followup, 'load', return_value=mismatched):
            with self.assertRaisesRegex(RuntimeError, 'Frozen code version differs'):
                reader_followup.verify()

    def test_offline_reanalysis_rejects_changed_frozen_data(self):
        manifest = deepcopy(reader_followup.load(reader_followup.OUT / 'freeze_manifest.json'))
        name = next(iter(manifest['source_sha256']))
        manifest['source_sha256'][name] = '0' * 64
        with patch.object(reader_followup, 'load', return_value=manifest):
            with self.assertRaisesRegex(RuntimeError, 'Frozen file changed'):
                reader_followup.verify(check_code=False)


if __name__=='__main__':
    unittest.main()
