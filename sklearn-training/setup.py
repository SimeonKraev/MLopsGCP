from setuptools import find_packages
from setuptools import setup


setup(name="trainer",
      version='0.1.0',
      packages=find_packages(),
      install_requires=['joblib', 'pandas', 'scikit-learn==1.0.2', 'google-cloud'],
      include_package_data=True,
      description="My sklearn training application.")
