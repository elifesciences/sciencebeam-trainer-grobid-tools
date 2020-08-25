elifePipeline {
    node('containers-jenkins-plugin') {
        def commit

        stage 'Checkout', {
            checkout scm
            commit = elifeGitRevision()
            if (env.TAG_NAME) {
                version = env.TAG_NAME - 'v'
            } else {
                version = 'develop'
            }
        }

        stage 'Build and run tests', {
            try {
                sh "make IMAGE_TAG=${commit} REVISION=${commit} ci-build-and-test"

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

        elifeTagOnly { tag ->
            stage 'Push release image', {
                def image = DockerImage.elifesciences(this, 'sciencebeam-trainer-grobid-tools', commit)
                image.tag('latest').push()
                image.tag(version).push()
            }
        }
    }
}
