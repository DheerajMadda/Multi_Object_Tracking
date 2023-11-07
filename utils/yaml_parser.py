import os
import yaml
from easydict import EasyDict

class YamlParser(EasyDict):
    """
    This class parses the yaml file
    """
    def __init__(self, config_dict=None, config_file=None):
        """
        This is the initialization method

        Parameters
        ----------
        config_dict : [None, dict]
            A None object or a dictionary of key-value pairs
        config_file : [None, str]
            A None object or a path to the yaml file

        """

        if config_dict is None:
            config_dict = {}

        if config_file is not None:
            assert(os.path.isfile(config_file))
            with open(config_file, 'r') as f:
                dict_ = yaml.load(f.read(), Loader=yaml.FullLoader)
                config_dict.update(dict_)

        super(YamlParser, self).__init__(config_dict)

    def merge_from_file(self, config_file):
        """
        This method updates the attributes provided a yaml file

        Parameters
        ----------
        config_file : str
            A path to the yaml file

        Returns
        -------
        None
        
        """

        assert(os.path.isfile(config_file))
        with open(config_file, 'r') as f:
            dict_ = yaml.load(f.read(), Loader=yaml.FullLoader)
            self.update(dict_) 

    def merge_from_dict(self, config_dict):
        """
        This method updates the attributes provided a dictionary

        Parameters
        ----------
        config_dict : [None, dict]
            A dictionary of key-value pairs

        Returns
        -------
        None
        
        """
        self.update(config_dict)
