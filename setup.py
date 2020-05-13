import setuptools


def readme():
    with open('README.md') as f:
        return f.read()

setuptools.setup(name='gasur',
                 version='0.0.0',
                 description='A package for Guidance, Navigation, and Control (GNC) for Autonomous Vehicles.',
                 long_description=readme(),
                 url='https://github.com/drjdlarson/gncpy',
                 author='Laboratory for Autonomy GNC and Estimation Research (LAGER)',
                 author_email='',
                 license='MIT',
                 packages=setuptools.find_packages('gncpy'),
                 package_dir={"": "gncpy"},
                 install_requires=['numpy', 'scipy'],
                 tests_require=['pytest', 'numpy'],
                 include_package_data=True,
                 zip_safe=False)
