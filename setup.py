"""
Setup script for AgroMRV System
Installation and packaging configuration
"""

from setuptools import setup, find_packages
from pathlib import Path
import os

# Read README file
README_PATH = Path(__file__).parent / "README.md"
long_description = README_PATH.read_text(encoding='utf-8') if README_PATH.exists() else ""

# Read requirements
REQUIREMENTS_PATH = Path(__file__).parent / "requirements.txt"
if REQUIREMENTS_PATH.exists():
    with open(REQUIREMENTS_PATH, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    requirements = [
        'streamlit>=1.28.0',
        'pandas>=1.5.0',
        'numpy>=1.24.0',
        'scikit-learn>=1.3.0',
        'plotly>=5.15.0',
        'scipy>=1.11.0',
        'python-dateutil>=2.8.0'
    ]

# Package metadata
setup(
    name="agromrv-system",
    version="1.0.0",
    author="AgroMRV Team",
    author_email="team@agromrv.com",
    description="AI-Powered Agricultural Monitoring, Reporting & Verification System for Indian Smallholder Farmers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/agromrv/agromrv-system",
    project_urls={
        "Bug Reports": "https://github.com/agromrv/agromrv-system/issues",
        "Source": "https://github.com/agromrv/agromrv-system",
        "Documentation": "https://agromrv.readthedocs.io/",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Office/Business :: Financial :: Accounting",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.0.0',
        ],
        'docs': [
            'sphinx>=5.0.0',
            'sphinx-rtd-theme>=1.2.0',
            'myst-parser>=0.18.0',
        ],
        'reporting': [
            'reportlab>=4.0.0',
            'openpyxl>=3.1.0',
            'jinja2>=3.1.0',
        ]
    },
    entry_points={
        'console_scripts': [
            'agromrv=run_app:main',
            'agromrv-dashboard=run_app:main',
        ],
    },
    include_package_data=True,
    package_data={
        'app': [
            'dashboard/templates/*',
            'data/samples/*',
            'static/*',
        ],
    },
    keywords=[
        'agriculture', 'mrv', 'monitoring', 'reporting', 'verification',
        'carbon', 'emissions', 'climate', 'sustainability', 'ipcc',
        'ai', 'machine-learning', 'blockchain', 'smallholder', 'farmers',
        'india', 'nabard', 'climate-finance'
    ],
    zip_safe=False,
)