pipeline {
    agent any

    options {
        skipDefaultCheckout(true)
        timestamps()
        disableConcurrentBuilds(abortPrevious: true)
    }

    environment {
        NOTIFY_EMAIL = 'emanuelacag@gmail.com'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Validate structure') {
            steps {
                sh '''
                    set -e
                    for p in \
                        mlops_pipeline/src \
                        mlops_pipeline/src/ft_engineering.py \
                        mlops_pipeline/src/model_training.py \
                        mlops_pipeline/src/model_deploy.py \
                        mlops_pipeline/src/model_evaluation.py \
                        mlops_pipeline/src/model_monitoring.py \
                        mlops_pipeline/src/config.json \
                        requirements.txt \
                        README.md
                    do
                        if [ ! -e "$p" ]; then
                            echo "Missing required path: $p" >&2
                            exit 1
                        fi
                    done
                    echo "Structure validation OK"
                '''
            }
        }
    }

    post {
        always {
            script {
                def body = """Pipeline: ${env.JOB_NAME}
Build: #${env.BUILD_NUMBER}
Result: ${currentBuild.currentResult}
URL: ${env.BUILD_URL}
Console: ${env.BUILD_URL}console
"""
                try {
                    // Email Extension Plugin: https://plugins.jenkins.io/email-ext/
                    // Uses SMTP from Manage Jenkins → Configure System → Extended E-mail Notification
                    emailext(
                        to: env.NOTIFY_EMAIL,
                        subject: "[Jenkins] ${env.JOB_NAME} #${env.BUILD_NUMBER} - ${currentBuild.currentResult}",
                        body: body,
                    )
                } catch (Throwable e) {
                    echo "Could not send email (configure SMTP in Extended E-mail Notification, not localhost:25): ${e.message}"
                }
            }
        }
    }
}
