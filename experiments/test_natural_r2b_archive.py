"""Archive and agreement tests using temporary synthetic fixtures only."""
import hashlib
import io
from pathlib import Path
import tarfile
import tempfile
import unittest

from test_natural_r2b_input_guards import fixture
from verify_archived_natural_r2b import expected_names, extract_verified, compare_adjudicators
from verify_natural_r2b_raw import SEEDS


class ArchiveGuards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='TEST_ONLY_archive_')
        self.root = Path(self.tmp.name)
        self.src = self.root/'fixtures'; self.src.mkdir()
        self.dst = self.root/'extracted'; self.dst.mkdir()
        for seed in SEEDS: fixture(self.src, seed)
        self.archive = self.root/'fixture.tar.gz'
        with tarfile.open(self.archive,'w:gz') as tar:
            for name in sorted(expected_names()): tar.add(self.src/name, arcname=name)
        self.manifest = {'archive_sha256':hashlib.sha256(self.archive.read_bytes()).hexdigest(),
                         'members':{name:hashlib.sha256((self.src/name).read_bytes()).hexdigest()
                                    for name in expected_names()}}
    def tearDown(self):self.tmp.cleanup()
    def test_allowlisted_regular_files_extract(self):
        extract_verified(self.archive,self.manifest,self.dst)
        self.assertEqual({p.name for p in self.dst.iterdir()},expected_names())
    def test_archive_hash_rejects_modification(self):
        self.archive.write_bytes(self.archive.read_bytes()+b'x')
        with self.assertRaises(ValueError):extract_verified(self.archive,self.manifest,self.dst)
    def test_member_hash_rejects_modification(self):
        key=sorted(expected_names())[0]
        self.manifest['members'][key]='0'*64
        with self.assertRaises(ValueError):extract_verified(self.archive,self.manifest,self.dst)
    def test_traversal_member_rejected(self):
        with tarfile.open(self.archive,'w:gz') as tar:
            info=tarfile.TarInfo('../outside'); info.size=1
            tar.addfile(info,io.BytesIO(b'x'))
        self.manifest['archive_sha256']=hashlib.sha256(self.archive.read_bytes()).hexdigest()
        with self.assertRaises(ValueError):extract_verified(self.archive,self.manifest,self.dst)
        self.assertFalse((self.root/'outside').exists())
    def test_verdict_disagreement_rejected(self):
        with self.assertRaises(ValueError):compare_adjudicators({'verdict':'FAIL'},{'verdict':'PASS'})
    def test_numeric_disagreement_rejected(self):
        raw={'verdict':'PASS','favorable_primary_signs':10,'mean_saving_vs_TJA':.1}
        original={'verdict':'PASS','n_favorable_primary_sign':10,'mean_saving_vs_TJA':.2}
        with self.assertRaises(ValueError):compare_adjudicators(raw,original)


if __name__=='__main__':unittest.main()
