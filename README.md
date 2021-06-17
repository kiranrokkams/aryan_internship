# Introduction

This repo is a collection of reference code for solving frequent usecases and projects.


# Pre-requisites

* Ensure you have `Miniconda` installed and can be run from your shell. If not, download the installer for your platform here: https://docs.conda.io/en/latest/miniconda.html

     **NOTE**
     
     * If you already have `Anaconda` installed, pls. still go ahead and install `Miniconda` and use it for development. 
     * If `conda` cmd is not in your path, you can configure your shell by running `conda init`. 


* Ensure you have `git` installed and can be run from your shell

     **NOTE**
     
     * If you have installed `Git Bash` or `Git Desktop` then the `git` cli is not accessible by default from cmdline. 
       If so, you can add the path to `git.exe` to your system path. Here are the paths on a recent setup
      
```
        %LOCALAPPDATA%\Programs\Git\git-bash.exe
        %LOCALAPPDATA%\GitHubDesktop\app-<ver>\resources\app\git\mingw64\bin\git.exe
```


* Ensure `GITHUB_OAUTH_TOKEN` is added as an environment variable by running the following command in respective terminals.
     * Windows Powershell
     ```
           $env:GITHUB_OAUTH_TOKEN="<token>" 
     ```
     * Windown CMD
     ```
           set GITHUB_OAUTH_TOKEN="<token>" 
     ```
     * Linux Shell
     ```
           export GITHUB_OAUTH_TOKEN="<token>" 
     ```


     **NOTE**
     
     * If you dont have a personal access token, Please follow the instructions described [here](https://docs.github.com/en/free-pro-team@latest/github/authenticating-to-github/creating-a-personal-access-token) and generate one.
      
```
        %LOCALAPPDATA%\Programs\Git\git-bash.exe
        %LOCALAPPDATA%\GitHubDesktop\app-<ver>\resources\app\git\mingw64\bin\git.exe
```


* Ensure [invoke](http://www.pyinvoke.org/index.html) tool and pyyaml are installed in your `base` `conda` environment. If not, run

```
(base):~$ pip install invoke
(base):~$ pip install pyyaml
```


# Getting started

* Clone the current repo to some local folder and switch to the root folder
* A collection of workflow automation tasks can be seen as follows
```
(base):~/code-templates$ inv -l
```

* Setup a development environment by running:
```
(base):~/code-templates$ inv dev.setup-env
```

The above command should create a conda python environment named `ta-lib-dev` and install the code in the current repository along with all required dependencies.

* Add reporting packages:
```
(base):~/code-templates$ conda activate ta-lib-dev
(ta-lib-dev):~/code-templates$ conda install xlrd==1.2.0 XlsxWriter==1.4.0 python-pptx==0.6.19
```

In case of installation issues, remove and recreate. You can remove by running:
```
(base):~/code-templates$ conda remove --name ta-lib-dev --all -y
```

# Testing

## Testing .py code

### Format QC

Use the inv task `test.qc` for fixing code standard issues

```
(ta-lib-dev):~/code-templates$ inv test.qc
```

- If you see `I001 isort` or `BLK100` issues, use this to autoformat and resolve these `inv test.qc`
- If you are not able to resolve any issues reach code templates team.


### Code testing

- All the test cases will go into `tests\` folder.
- Look at the existing test scripts to learn writing test cases. More can be found in **developer guide** on readthedocs.


# Documentation

Use proper doc strings in all public classes & functions.