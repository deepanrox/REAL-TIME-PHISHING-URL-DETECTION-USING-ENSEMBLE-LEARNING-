from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="phishing-websites-detection",
    version="1.0.0",
    author="[YOUR_NAME_HERE]",  # TODO: Replace with your actual name
    author_email="[YOUR_EMAIL_HERE]",  # TODO: Replace with your actual email
    description="A machine learning-based system to detect phishing websites using URL features",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/[YOUR_USERNAME]/phishing-websites-detection",  # TODO: Replace with your GitHub username
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    entry_points={
        "console_scripts": [
            "phishing-detector=phishing_detector:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.csv", "*.pkl", "*.joblib"],
    },
) 