def make_test_artifact(artifact_type):
    class _A(artifact_type):
        def _get_path(self):
            return self.uri
        @property
        def path(self):
            return self.uri
    return _A
