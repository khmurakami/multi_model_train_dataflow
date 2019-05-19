import setuptools

setuptools.setup(
    name="multi_model_train_dataflow",
    version="0.0.3",
    author="Kalani Murakami",
    author_email="kalanimurakami1218@gmail.com",
    description="Train Multiple Models using Dataflow",
    packages=['multi_model_train'],
    install_requires=["tensorflow", "apache-beam", "scikit-learn"],
    license="MIT",
    url="https://github.com/khmurakami/multi_model_train_dataflow"
)
