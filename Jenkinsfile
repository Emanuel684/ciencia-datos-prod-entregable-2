pipeline {
    agent any

    options {
        skipDefaultCheckout(true)
        timestamps()
    }

    // Polling when Jenkins is not reachable from GitHub (typical local Docker).
    // For instant builds on push/merge, configure GitHub webhook + Multibranch scan (see JENKINS.md).
    triggers {
        pollSCM('H/15 * * * *')
    }

    stages {
        // stage('Checkout') {
        //     steps {
        //         script {
        //             def extensions = (scm.extensions ?: []) + [
        //                 [
        //                     $class             : 'SubmoduleOption',
        //                     disableSubmodules  : false,
        //                     parentCredentials  : true,
        //                     recursiveSubmodules: true,
        //                     reference          : '',
        //                     trackingSubmodules : false,
        //                 ],
        //             ]
        //             checkout([
        //                 $class           : 'GitSCM',
        //                 branches         : scm.branches,
        //                 extensions       : extensions,
        //                 userRemoteConfigs: scm.userRemoteConfigs,
        //             ])
        //         }
        //     }
        // }

        // stage('Verify structure') {
        //     steps {
        //         sh '''
        //             set -e
        //             for p in \
        //                 ci-p03-python-etl/mlops_pipeline/src \
        //                 ci-p03-python-etl/mlops_pipeline/src/model_training.py \
        //                 ci-p03-python-etl/mlops_pipeline/src/model_deploy.py \
        //                 ci-p03-python-etl/mlops_pipeline/src/ft_engineering.py \
        //                 docker-compose.yml \
        //                 Jenkinsfile
        //             do
        //                 if [ ! -e "$p" ]; then
        //                     echo "Missing required path: $p" >&2
        //                     exit 1
        //                 fi
        //             done
        //             echo "Structure check OK"
        //         '''
        //     }
        // }

        stage('Clonar repo') {
            steps {
                git 'https://github.com/tu-usuario/tu-repo.git'
            }
        }

        stage('Instalar dependencias') {
            steps {
                sh 'npm install'  // o pip install -r requirements.txt
            }
        }

        stage('Tests') {
            steps {
                sh 'npm test'  // o pytest
            }
        }
    }

    // post {
    //     always {
    //         script {
    //             def credId = 'notify-webhook-url'
    //             try {
    //                 withCredentials([string(credentialsId: credId, variable: 'NOTIFY_WEBHOOK_URL')]) {
    //                     def payload = groovy.json.JsonOutput.toJson([
    //                         status   : currentBuild.currentResult,
    //                         job      : env.JOB_NAME,
    //                         build    : env.BUILD_NUMBER,
    //                         buildUrl : env.BUILD_URL,
    //                     ])
    //                     writeFile file: 'jenkins-notify-payload.json', text: payload
    //                     sh '''
    //                         if [ -z "${NOTIFY_WEBHOOK_URL}" ]; then
    //                             echo "Empty NOTIFY_WEBHOOK_URL; skip HTTP notification."
    //                             exit 0
    //                         fi
    //                         set -e
    //                         curl -sS -f -X POST \
    //                             -H "Content-Type: application/json" \
    //                             --data-binary @jenkins-notify-payload.json \
    //                             "${NOTIFY_WEBHOOK_URL}"
    //                     '''
    //                 }
    //             } catch (Throwable e) {
    //                 echo "Webhook notification skipped or failed (optional credential '${credId}'): ${e.message}"
    //             }
    //         }
    //     }
    // }
}
