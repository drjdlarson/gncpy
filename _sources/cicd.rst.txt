CI/CD Pipeline
==============

The goal of Continuous Integration and Continuous Development (CI/CD) is to be constantly running tests and checks upon every push to the repo. This project makes use of Github actions to achieve this, but other technologies can be used such as Travis CI or Jenkins. Two main workflows are setup, once that runs on every push and another that is designed to automate releases.

The first that runs on every push is designed to run the tests on multiple versions of python on multiple operating systems. If the tests fail, you can look at the workflow log to see the output from the failing test cases.

The second workflow is designed to run when pushing a new version tag to automate creating a new release. It will build the container images and publish them to the Github Container Registry, build and run the unit tests, build the documentation, push the documentation to the gh-pages branch to update the website, and automatically package the release and add a commit message to the release that summarizes all the commits since the last release.


Github Actions Quick Guide
--------------------------
The following is a quick reference to Github actions. Each action, or workflow, is defined within a :code:`.github/workflow/*.yml` file. These files follow the YAML syntax and can define multiple jobs which can run in parallel or in sequence. The file defines triggers that control when each workflow gets executed. All of these workflows execute based on events that happen on the remote (i.e. if a workflow is triggered on a version tag, making a version tag on your local system with :code:`git tag` won't trigger the action, but pushing that to github with a :code:`git push` will cause it to be triggered since github will now see the tag). Additionally, the workflows run on Github's servers NOT your local machine.

Each job within a workflow starts with a fresh system (i.e. a clean diretory) and can be defined to run on a different operating system (as defined by the runner). A given job is made up of multiple steps, a step can either run commands in the runner's terminal or use a predefined action which may have additional options. A typicaly job will need a step to checkout the repo at the current state that triggered the action. The steps within a job run sequentially, whereas the jobs themselves normally run in parallel. The jobs can be made to run in sequence by defining the :code:`needs` tag within the job and giving the id of a prior job.