from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    description = f.read()

with open('requirements.txt', 'r') as f:
    req_list = f.read().splitlines()

setup(
    name='veg_seg',
    version='0.1.0',
    description='A package for classifying vegetation from lidar point clouds',
    author='Phillipe Wernette', 
    author_email='pwernett@msu.edu',
    packages=find_packages(),
    install_requires=[
        "matplotlib",
        "ipython",
        "graphviz",
        "scikit-learn",
        "pandas",
        "geopandas",
        "scipy",
        "tqdm",
        "pydot",
        "laspy",
        "lazrs",
        "numpy==1.26.4;platform_system=='Windows'",
        "tensorflow[and-cuda]==2.17;platform_system=='Linux'",
        "tensorflow-gpu==2.10;platform_system=='Windows'",
    ],
    entry_points={
        'console_scripts': [
            'vegfilter=veg_seg.main:veg_filter',
            'vegreclass=veg_seg.main:veg_reclass',
            'vegtrain=veg_seg.main:veg_train',
        ],
    },
    long_description=description,
    long_description_content_type='text/markdown',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: GIS',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
    ],
)