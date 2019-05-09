elifePipeline {
    node('containers-jenkins-plugin') {
        def commit

        stage 'Checkout', {
            checkout scm
            commit = elifeGitRevision()
        }

        stage 'Build and run tests', {
            try {
                sh "make ci-build-and-test"

                echo "Checking revision label..."
                def image = DockerImage.elifesciences(this, 'sciencebeam-trainer-grobid-tools', commit)
                echo "Reading revision label of image: ${image.toString()}"
                def actualRevision = sh(
                    script: "./ci/docker-read-local-label.sh ${image.toString()} org.opencontainers.image.revision",
                    returnStdout: true
                ).trim()
                echo "revision label: ${actualRevision} (expected: ${commit})"
                assert actualRevision == commit
            } finally {
                sh "make ci-clean"
            }
        }

        elifeMainlineOnly {
            stage 'Merge to master', {
                elifeGitMoveToBranch commit, 'master'
            }

            stage 'Push unstable image', {
                def image = DockerImage.elifesciences(this, 'sciencebeam-trainer-grobid-tools', commit)
                def unstable_image = image.addSuffixAndTag('_unstable', commit)
                unstable_image.tag('latest').push()
                unstable_image.push()
            }
        }
    }
}
