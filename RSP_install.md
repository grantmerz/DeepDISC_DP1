### DeepDISC install instructions for the RSP

Follow the instructions below to set up an environment with DeepDISC and the LSST stack


1. Clone the deepdisc repo:  
    `git clone https://github.com/grantmerz/deepdisc`  
   

2. Install deepdisc prerequisites.  The current environment has a lot of what we need already, so just install timm:  
    `pip install timm --user`

   
3. Install detectron2:  
    `python -m pip install --user 'git+https://github.com/facebookresearch/detectron2.git'`


4.   Install detectron2:  
    `python -m pip install --user -e detectron2`

    You will likely need to include the --no-build-isolation flag when pip installing detectron2. This is due to newer versions of pip isolating package builds. Detectron2 requires pytorch to build, so it must be seen in the current environment. You may have other issues with buulding detectron2. Check their installation documentation for some common errors and fixes.
 
5. Install deepdisc:  
    `cd deepdisc`  
    `pip install .`
--------------------------------------------------
Optional for models that use dustmaps

6. Reset dustmaps config and install the sfd dustmap:  
    `python -c "from dustmaps.config import config and import config.reset()"`


   `python -c "from dustmaps.config import config; config['data_dir'] = '/home/{$USER}/DATA/dustmaps'"`


   `python -c "import dustmaps.sfd; dustmaps.sfd.fetch()"`
       
