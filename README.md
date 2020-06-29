# gncpy
A python package for guidance, navigation, and control (GNC) algorithms developed by the Laboratory for Autonomy, GNC, and Estimation Research (LAGER) at the University of Alabama (UA).

## Unit testing
Unit tests make use **pytest** for the test runner, and tox for automation. The test scripts are located within the **test/** sub-directory.
The tests can be run through a command line with python 3 and tox installed by running
`tox -e test`

Specific tests can be run by passing keywords such as
`tox -e test -- -k guidance`

To run tests marked as slow, pass the `--runslow` option,
`tox -e test -- --runslow`

## Building documentation
The documentation uses sphinx and autodoc to pull docstrings from the code. This process is run through a command line that has python 3 and tox installed. The built documentation is in the **docs/build/** sub-directory.
The HTML version of the docs can be built using the following command:
`tox -e docs -- html`

Then they can be viewed by opening **docs/build/html/index.html**
