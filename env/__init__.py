"""
store env param under different occasion
"""
import yaml

MODES = ['preset', 'test', 'finetuned']
ENV_REGISTRY = ['mpe', 'smac']
MAP_REGISTRY = {'mpe': ['reference', 'risk_ref', 'tag'],
                'smac': ['3m', '8m', ]}


def config_env(mode='preset', env='mpe', map='reference'):
    assert mode in MODES, f"Undefined mode <{mode}>"
    assert env in ENV_REGISTRY, f"Undefined env <{env}>"
    assert map in MAP_REGISTRY[env], f"Undefined map <{map}> of env <{env}>"
    config_path = f'./env/{mode}/{env}_{map}.yaml'
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    return config
