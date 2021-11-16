import setuptools
import numpy

with open("README.org", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="model_selection_breakpoints",
    version="2021.11.15",
    author="Toby Dylan Hocking",
    author_email="Toby.Hocking@nau.edu",
    description="Exact breakpoints in model selection function",
    long_description=long_description,
    long_description_content_type="text/plain",
    url="https://github.com/tdhock/model_selection_breakpoints",
    install_requires=['numpy'],
    extras_require={
        'test': ['pytest']
    },
    ext_modules=[setuptools.Extension('model_selection_breakpoints',
                                      ['interface.c', 'modelSelectionFwd.c'],
                                      include_dirs=[numpy.get_include()])],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: C",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires='>=3.6',
)
