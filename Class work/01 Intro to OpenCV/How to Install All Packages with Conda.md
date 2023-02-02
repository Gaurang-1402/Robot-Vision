I recommend using Anaconda to install Jupyter notebook and OpenCV. Anaconda can help you manage different environments. Imagine you have 2 projects going. One needs software A which requires software B version 1. Another project where you need software C which require software B version 2 and not backward compatible. Now you wish that you can have two computers and install software B on both computer with different versions. Anaconda helps you achieve this without actually getting two computers. 


Generally speaking, getting to know how to manage environments is good down the road as a Python developer. But for the purpose of this class, it's just an easier way for everybody to have the exact same setup and software installed, so I can better help you.
1. Install Anaconda following their [official documentation](https://docs.continuum.io/anaconda/install/).

2. If you follow their documentation step-by-step, when you open your terminal, you will most likely need to run `conda init` to initial conda. Close and restart your terminal after this. (Sometimes you don't need to, it depends on the terminal and the operating system you use)

3. If step 2 is successful, there should be a `(base)` appearing to the left in your terminal. (If after installing conda you already have this, then you can skip step 2)

4. In the same folder that has the `environment.yml` file, run `conda env create -f environment.yml`. This should create a virtual environment that has all the required packages we need for this class at this stage.
5. Run `conda activate vision_class` to activate the environment we just created.

5. Run `jupyter notebook` in the same folder to use the `.ipynb` file I shared.

If you encounter any problem during the setup, please don't hesitate to DM me on Slack or schedule a zoom OH with me over Slack