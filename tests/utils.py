import typing


def make_test_artifact(ARTIFACT_TYPE: typing.Type):

    class TestArtifact(ARTIFACT_TYPE):
        def _get_path(self):
            return super()._get_path() or self.uri

    return TestArtifact
