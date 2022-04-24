import itertools
import setuptools

# environment specific requirements
extras = {
    'reinforcement-learning': ["pygame>=2.1.2", "gym>=0.19"],
    'simple-multirotor': ["ruamel.yaml>=0.17.21"]
    }  # NOQA

extras['all'] = list(itertools.chain.from_iterable(map(lambda group: extras[group],
                                                       extras.keys())))


def readme():
    with open('README.md') as f:
        return f.read()


setuptools.setup(name='gncpy',
                 version='0.0.0',
                 description='A package for Guidance, Navigation, and Control (GNC) algorithms.',
                 # long_description=readme(),
                 url='https://github.com/drjdlarson/gncpy',
                 author='Laboratory for Autonomy GNC and Estimation Research (LAGER)',
                 author_email='',
                 license='MIT',
                 packages=setuptools.find_packages(),
                 install_requires=[
                     'numpy',
                     'scipy',
                     'matplotlib'
                 ],
                 extras_require=extras,
                 tests_require=['pytest', 'numpy'],
                 include_package_data=True,
                 zip_safe=False)
