"""Real tiny Git repositories, stdlib only; never stage/commit the project.

The repository baseline is synthetic. A wrapper translates ONLY the fixed
production baseline token into that fixture's baseline hash; the actual helper
still receives/uses the pinned P4-A constant, never dynamic HEAD.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PINNED = 'e60eb5a96662f9e448df571e5e0e495bd43ba737'
LIFECYCLE = ROOT / 'tests/test_agent_sse_lifecycle_isolated.py'
POC = ROOT / 'tests/test_asgi_streaming_poc_isolated.py'


def helpers(path, wanted):
    """Extract the actual audited constants/functions; no application imports."""
    tree = ast.parse(path.read_text(), filename=path.name)
    nodes = []
    found = set()
    for node in tree.body:
        names = ({node.name} if isinstance(node, ast.FunctionDef) else
                 {t.id for t in node.targets if isinstance(t, ast.Name)}
                 if isinstance(node, ast.Assign) else set())
        if names & wanted:
            nodes.append(node)
            found.update(names & wanted)
    if found != wanted:
        raise AssertionError('Missing real freeze helper/constant')
    scope = {'hashlib': hashlib, 'json': json, 'os': os, 'subprocess': subprocess}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path.name, 'exec'), scope)
    return types.SimpleNamespace(**scope)


scope = helpers(LIFECYCLE, {'BASELINE', 'PRODUCTION_FILES', 'REQUIRED_ADDITIONS',
    'OPTIONAL_REPAIR_ADDITIONS', 'git_scope_records', 'assert_authorized_change_scope'})
content = helpers(POC, {'BASELINE', 'PRODUCTION_FILES', 'FROZEN_MANIFEST_SHA256',
    'AUTHORIZED_TEST_REPAIRS', 'frozen_production_diff', 'assert_protected_p4b_files'})
RAW_GIT = subprocess.check_output


def git_env():
    # No inherited GIT_DIR/INDEX/alternates/config, no global config read/write.
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
               GIT_TERMINAL_PROMPT='0')
    return env


def deny_network(event, args):
    if event in {'socket.connect', 'socket.bind', 'socket.getaddrinfo'}:
        raise AssertionError('Network is forbidden in freeze fixtures')


sys.addaudithook(deny_network)


class GitFixture:
    def __init__(self, originals, approved):
        self.temp = tempfile.TemporaryDirectory(prefix='movieagent-freeze-r0-', dir='/tmp')
        self.root = Path(self.temp.name) / 'repo'
        self.root.mkdir()
        self.hooks = Path(self.temp.name) / 'empty-hooks'
        self.hooks.mkdir()
        self.env = git_env()
        self.config = ['-c', 'core.hooksPath=' + str(self.hooks),
            '-c', 'user.name=Isolated Fixture', '-c', 'user.email=fixture@example.invalid',
            '-c', 'commit.gpgSign=false', '-c', 'core.autocrlf=false']
        self.calls = []
        self.git('init', '--quiet')
        baseline = dict(originals)
        baseline['myapp/unrelated.py'] = b'# synthetic control only\n'
        baseline['tests/existing_control.py'] = b'# synthetic control only\n'
        baseline['.gitignore'] = b'.env\n'
        for name, data in baseline.items():
            self.write(name, data)
        self.git('add', '--', *sorted(baseline))
        self.git('commit', '--quiet', '-m', 'synthetic fixture baseline')
        self.baseline = self.git('rev-parse', 'HEAD').decode().strip()
        self.approved = approved
        for name, data in approved.items():
            self.write(name, data)

    def write(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def git(self, *args):
        return RAW_GIT(['git', *self.config, *args], cwd=self.root, env=self.env,
                       timeout=5, stderr=subprocess.PIPE)

    def state(self, mode):
        if mode != 'unstaged':
            self.git('add', '--', *sorted(self.approved))
        if mode == 'committed':
            self.git('commit', '--quiet', '-m', 'synthetic approved freeze')

    def translate(self, command, **kwargs):
        if Path(kwargs.get('cwd', '.')).resolve() != self.root.resolve():
            raise AssertionError('Fixture helper tried to use another Git repository')
        self.calls.append(tuple(command))
        args = [self.baseline if value == PINNED else
                self.baseline + value[len(PINNED):] if value.startswith(PINNED + ':') else value
                for value in command[1:]]
        # This is real Git over a separate .git, not fabricated Git output.
        return self.git(*args)

    def check(self, *, content_check=True, scope_check=True):
        with patch.object(subprocess, 'check_output', side_effect=self.translate):
            if scope_check:
                scope.assert_authorized_change_scope(self.root)
            if content_check:
                content.assert_protected_p4b_files(self.root)

    def close(self):
        self.temp.cleanup()  # Only the explicit isolated TemporaryDirectory.


class FreezeStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if scope.BASELINE != PINNED or content.BASELINE != PINNED:
            raise AssertionError('Actual production freeze baseline must be pinned')
        cls.originals = {name: RAW_GIT(['git', 'show', PINNED + ':' + name],
                                     cwd=ROOT, timeout=5) for name in scope.PRODUCTION_FILES}
        cls.approved = {name: (ROOT / name).read_bytes()
                        for name in scope.PRODUCTION_FILES | scope.REQUIRED_ADDITIONS}

    def fixture(self, state='unstaged'):
        fixture = GitFixture(self.originals, self.approved)
        self.addCleanup(fixture.close)
        fixture.state(state)
        return fixture

    def reject_scope(self, fixture):
        with self.assertRaises(AssertionError):
            fixture.check(content_check=False)

    def reject_content(self, fixture):
        with self.assertRaises(AssertionError):
            fixture.check(scope_check=False)

    def test_unstaged_passes(self):
        self.fixture().check()

    def test_staged_passes_despite_empty_default_diff(self):
        f = self.fixture('staged')
        self.assertEqual(f.git('diff', '--binary'), b'')
        f.check()

    def test_committed_passes_despite_empty_head_diff(self):
        f = self.fixture('committed')
        self.assertEqual(f.git('diff', 'HEAD', '--binary'), b'')
        self.assertNotEqual(f.baseline, f.git('rev-parse', 'HEAD').decode().strip())
        f.check()

    def test_fixed_baseline_diff_hash_identical_in_all_states(self):
        expected = json.loads(self.approved['docs/agent_backend/p4b1/P4B_FROZEN_SHA256.json'])['p4b_diff_sha256']
        for state in ('unstaged', 'staged', 'committed'):
            with self.subTest(state=state):
                f = self.fixture(state)
                with patch.object(subprocess, 'check_output', side_effect=f.translate):
                    digest = hashlib.sha256(content.frozen_production_diff(f.root)).hexdigest()
                self.assertEqual(digest, expected)
                self.assertIn(PINNED, f.calls[-1])

    def test_head_advance_preserves_pinned_baseline(self):
        f = self.fixture('committed')
        f.git('commit', '--quiet', '--allow-empty', '-m', 'synthetic HEAD advance')
        f.check()
        diff_commands = [call for call in f.calls if call[1] == 'diff' and '--binary' in call]
        self.assertTrue(diff_commands)
        self.assertTrue(all(PINNED in command and 'HEAD' not in command for command in diff_commands))

    def test_replacing_pinned_constant_with_dynamic_head_is_rejected(self):
        f = self.fixture('committed')
        with patch.dict(content.assert_protected_p4b_files.__globals__, {'BASELINE': 'HEAD'}):
            self.reject_content(f)

    def test_unknown_production_change_rejected_in_all_states(self):
        for state in ('unstaged', 'staged', 'committed'):
            with self.subTest(state=state):
                f = self.fixture(state)
                f.write('myapp/unrelated.py', b'# synthetic unapproved modification\n')
                if state != 'unstaged': f.git('add', '--', 'myapp/unrelated.py')
                if state == 'committed': f.git('commit', '--quiet', '-m', 'synthetic bad production')
                self.reject_scope(f)

    def test_frozen_production_tamper_rejected_in_all_states(self):
        name = 'myapp/agent_views.py'
        for state in ('unstaged', 'staged', 'committed'):
            with self.subTest(state=state):
                f = self.fixture(state)
                f.write(name, self.approved[name] + b'\n# synthetic tamper\n')
                if state != 'unstaged': f.git('add', '--', name)
                if state == 'committed': f.git('commit', '--quiet', '-m', 'synthetic tamper')
                # Scope alone permits this path; content must still reject it.
                f.check(content_check=False)
                self.reject_content(f)

    def test_unknown_addition_rejected_in_all_states(self):
        name = 'docs/unapproved.md'
        for state in ('unstaged', 'staged', 'committed'):
            with self.subTest(state=state):
                f = self.fixture(state)
                f.write(name, b'synthetic unknown addition\n')
                if state != 'unstaged': f.git('add', '--', name)
                if state == 'committed': f.git('commit', '--quiet', '-m', 'synthetic unknown addition')
                self.reject_scope(f)

    def test_production_deletion_rejected_in_all_states(self):
        name = 'myapp/agent_views.py'
        for state in ('unstaged', 'staged', 'committed'):
            with self.subTest(state=state):
                f = self.fixture(state)
                (f.root / name).unlink()
                if state != 'unstaged': f.git('add', '--', name)
                if state == 'committed': f.git('commit', '--quiet', '-m', 'synthetic deletion')
                self.reject_scope(f)
                self.reject_content(f)

    def test_frozen_document_deletion_rejected_in_all_states(self):
        name = 'docs/agent_backend/P4B_GO_NO_GO.md'
        for state in ('unstaged', 'staged', 'committed'):
            with self.subTest(state=state):
                f = self.fixture(state)
                (f.root / name).unlink()
                if state != 'unstaged': f.git('add', '--', name)
                if state == 'committed': f.git('commit', '--quiet', '-m', 'synthetic document deletion')
                self.reject_scope(f)
                self.reject_content(f)

    def test_staged_bad_blob_cannot_hide_under_good_worktree(self):
        f = self.fixture('staged')
        name = 'myapp/agent_views.py'
        f.write(name, self.approved[name] + b'\n# staged tamper\n')
        f.git('add', '--', name)
        f.write(name, self.approved[name])
        self.reject_content(f)

    def test_committed_bad_blob_cannot_hide_under_restored_index_and_worktree(self):
        f = self.fixture('committed')
        name = 'myapp/agent_views.py'
        f.write(name, self.approved[name] + b'\n# committed tamper\n')
        f.git('add', '--', name)
        f.git('commit', '--quiet', '-m', 'synthetic bad blob')
        f.write(name, self.approved[name])
        f.git('add', '--', name)
        self.reject_content(f)

    def test_unknown_staged_addition_deleted_from_worktree_rejected(self):
        f = self.fixture('staged')
        f.write('unknown.md', b'synthetic\n')
        f.git('add', '--', 'unknown.md')
        (f.root / 'unknown.md').unlink()
        self.reject_scope(f)

    def test_unknown_head_addition_deleted_from_index_and_worktree_rejected(self):
        f = self.fixture('committed')
        f.write('unknown.md', b'synthetic\n')
        f.git('add', '--', 'unknown.md')
        f.git('commit', '--quiet', '-m', 'synthetic unknown HEAD file')
        (f.root / 'unknown.md').unlink()
        f.git('add', '--', 'unknown.md')
        self.reject_scope(f)

    def test_existing_nonproduction_file_modification_rejected(self):
        f = self.fixture()
        f.write('tests/existing_control.py', b'# synthetic forbidden test edit\n')
        self.reject_scope(f)

    def test_ignored_unknown_file_is_not_a_scope_escape(self):
        f = self.fixture()
        f.write('.env', b'SYNTHETIC_ONLY=1\n')
        self.reject_scope(f)

    def test_production_rename_rejected(self):
        f = self.fixture('committed')
        f.git('mv', 'myapp/agent_views.py', 'myapp/renamed.py')
        self.reject_scope(f)
        self.reject_content(f)

    def test_frozen_document_rename_rejected(self):
        f = self.fixture('committed')
        f.git('mv', 'docs/agent_backend/P4B_GO_NO_GO.md', 'docs/renamed.md')
        self.reject_scope(f)
        self.reject_content(f)

    def test_missing_untracked_required_test_rejected(self):
        f = self.fixture()
        (f.root / 'tests/test_asgi_streaming_poc_isolated.py').unlink()
        self.reject_scope(f)

    def test_symlink_instead_of_frozen_file_rejected(self):
        f = self.fixture()
        path = f.root / 'myapp/agent_views.py'
        path.unlink()
        path.symlink_to('unrelated.py')
        self.reject_scope(f)
        self.reject_content(f)

    def test_manifest_tampering_not_accepted(self):
        f = self.fixture()
        name = 'docs/agent_backend/p4b1/P4B_FROZEN_SHA256.json'
        manifest = json.loads(self.approved[name])
        manifest['p4b_diff_sha256'] = hashlib.sha256(b'').hexdigest()
        f.write(name, json.dumps(manifest).encode())
        self.reject_content(f)

    def test_authorized_test_revision_is_exact_not_unlimited(self):
        f = self.fixture()
        name = 'tests/test_agent_sse_lifecycle_isolated.py'
        f.write(name, self.approved[name] + b'\n# unexpected test change\n')
        self.reject_content(f)

    def test_original_frozen_document_content_still_required(self):
        f = self.fixture()
        name = 'docs/agent_backend/P4B_GO_NO_GO.md'
        f.write(name, self.approved[name] + b'\nmodified\n')
        self.reject_content(f)

    def test_exact_r0_optional_paths_are_allowed(self):
        f = self.fixture()
        for name in scope.OPTIONAL_REPAIR_ADDITIONS:
            f.write(name, b'# synthetic optional artifact\n')
        f.check()
        f.git('add', '--', *sorted(scope.OPTIONAL_REPAIR_ADDITIONS))
        f.check()
        f.git('commit', '--quiet', '-m', 'synthetic optional artifacts')
        f.check()

    def test_similarly_named_optional_file_not_allowed(self):
        f = self.fixture()
        f.write('docs/agent_backend/p4b2/UNAPPROVED.md', b'synthetic\n')
        self.reject_scope(f)

    def test_unknown_filename_with_newline_rejected(self):
        f = self.fixture()
        f.write('unknown\nname.md', b'synthetic\n')
        self.reject_scope(f)

    def test_mixed_staging_of_same_approved_content_passes(self):
        f = self.fixture()
        f.git('add', '--', 'myapp/agent_views.py', 'tests/test_agent_sse_lifecycle_isolated.py')
        f.check()

    def test_staged_production_mode_change_cannot_hide_under_restored_worktree(self):
        f = self.fixture('staged')
        name = 'myapp/agent_views.py'
        f.git('update-index', '--chmod=+x', '--', name)
        self.reject_content(f)

    def test_staged_bad_document_cannot_hide_under_good_worktree(self):
        f = self.fixture('staged')
        name = 'docs/agent_backend/P4B_GO_NO_GO.md'
        f.write(name, b'bad staged synthetic report')
        f.git('add', '--', name)
        f.write(name, self.approved[name])
        self.reject_content(f)

    def test_current_actual_project_freeze_and_scope_pass(self):
        # These are exclusively read-only Git commands in the actual worktree.
        scope.assert_authorized_change_scope(ROOT)
        content.assert_protected_p4b_files(ROOT)

    def test_helpers_have_no_dynamic_or_external_diff_configuration(self):
        self.assertEqual(scope.BASELINE, PINNED)
        self.assertEqual(content.BASELINE, PINNED)
        source = ast.get_source_segment(POC.read_text(),
            next(n for n in ast.parse(POC.read_text()).body
                 if isinstance(n, ast.FunctionDef) and n.name == 'frozen_production_diff'))
        self.assertIn('--no-ext-diff', source)
        self.assertIn('--no-textconv', source)
        self.assertIn('BASELINE', source)
        self.assertNotIn('HEAD', source)
        self.assertEqual(set(content.PRODUCTION_FILES), scope.PRODUCTION_FILES)


if __name__ == '__main__':
    if not sys.flags.no_site or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or any(
        os.environ.get(name) != '1' for name in
        ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')):
        raise SystemExit('CPU-only/no-site environment required')
    unittest.main(verbosity=2)
