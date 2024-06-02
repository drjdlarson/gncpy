GNCPy
=====

A python package for Guidance, Navigation, and Control (GNC) algorithms developed by the Laboratory for Autonomy, GNC, and Estimation Research (LAGER) at the University of Alabama (UA).

For using this package, simply clone the repository and then pip install the top level folder. It is recommended to install with the `-e` flag such that new updates can be applied automatically. For setting up a development environment to extend the package see the following sections.

.. contents:: Table of Contents
    :depth: 2
    :local:
    :backlinks: entry

..
    BEGIN LINKS INCLUDE

.. |Open in Dev Containers| image:: https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode
   :target: https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/drjdlarson/gncpy.git

.. |Test Status| image:: https://drjdlarson.github.io/gncpy/reports/junit/tests-badge.svg?dummy=8484744
    :target: https://drjdlarson.github.io/gncpy/reports/junit/junit.html

.. |Test Cov| image:: https://drjdlarson.github.io/gncpy/reports/coverage/coverage-badge.svg?dummy=8484744
    :target: https://drjdlarson.github.io/gncpy/reports/coverage/index.html

..
    END LINKS INCLUDE

|Open in Dev Containers| |Test Status| |Test Cov|

..
    BEGIN TOOLCHAIN INCLUDE

This project uses the pytest for developing and running the tests (with extensions for generating summary and coverage reports), tox automates setting up and running the test environment (as well as the documentation), Sphinx is used for documenting the code, and the Black formatter is used to auto format the python code. This project also follows `semantic versioning <https://semver.org/>`__ where any api breaking changes will increment the major version number. Additionally, a prebuilt docker container image is provided to get started with developing for this library. It contains all of the needed tools to compile the code, run the tests, and build the documentation. The docker container can be used within VS Code through their dev container extension to allow editing local files but compiling using the toolchain provided within the container.


Development Environment Setup
-----------------------------
It is recommended to use VS Code with the dev containers extension for developing. Development containers allow the full toolchain to be automatically setup on most any machine capable of running Docker. For information on dev-containers see `here <https://code.visualstudio.com/docs/devcontainers/containers>`__ for an overview, `here <https://stackoverflow.com/questions/71402603/vs-code-in-docker-container-is-there-a-way-to-automatically-install-extensions>`__ for auto installing extensions in the container
and `here <https://pspdfkit.com/blog/2020/visual-studio-code-cpp-docker/>`__ for an example setup. The provided dev container also has useful extensions installed to ease development.

To being, make sure VS Code and git are installed. Additionally, make sure docker is installed for your system (`Windows <https://docs.docker.com/desktop/install/windows-install/>`__, `Mac <https://docs.docker.com/desktop/install/mac-install/>`_, `Linux <https://docs.docker.com/engine/install/>`__). Next, install the dev containers extension within VS Code. Clone the repository locally on your computer, for windows it is recommended to clone it within your linux subsystem directory (e.g. a sub-directory of your linux home folder) to improve performance within the container (the linux directories on Windows can be accessed through the file browser by typing :code:`\\wsl$` in the address bar and clicking on your distro). Now open the repo folder within VS Code (for windows you may need to connect to the linux subsystem first). Then you should be prompted to open the folder in the container, click yes. If you are not prompted, you can go to the command palette and start typing "Open folder in container". Now your terminal within VS Code will be running commands within the container but the files your are editing/creating will be accessible from your local machine's file browser.

Note if you click the open in container button on the repo's page it will automatically open VS Code, open the container, and clone the repo for you. However, it will do this within a docker volume so the files are only accessible within the container (ie you can't view them through your local file browser).


Example Workflow
----------------
Once the repository is open in the container, you can edit files, run tests, and make commits just like normal. For example, after editing some files and adding some validation tests to run these tests you would simply call the following from the root of the repository.

.. code-block:: 

    tox

This will attempt to run the all the validation tests, except those marked as slow, on multiple versions of python. If the python version can not be found, it will be skipped.

After running tests, it may be helpful to check the documentation build locally to ensure code comments are being pulled correctly. This can be done with

.. code-block:: 

    tox -e clean_docs
    tox -e docs_html

to remove any existing documenation builds and generate the html version. The output is placed in **docs/build/html** and can be viewed by opening the **docs/build/html/index.html** file in your web browser.


Notes on tox
------------
Tox will automatically create virtual environements, install dependencies, install the package, and run some commands in the virtual environment. These are defined in the **tox.ini** file in the repository. If tox is called without specifying an envrionment, it will run all of the default environments. The available environments can be listed with

.. code-block:: 

    tox -av

and a specific environment run by calling

.. code-block:: 

    tox -e ENV

where :code:`ENV` is replaced with the environment name. To pass positional arguments into the commands run within the tox environment you must use :code:`--` after the environment name but before the positional arguments. For example to run validation tests using Python 3.9 and pass the :code:`--runslow` option to pytest you would call :code:`tox -e py39-validation_test -- --runslow`.

Note, all tox commands must be run from the root of the repository because this is where the **tox.ini** file lives.

..
    END TOOLCHAIN INCLUDE
