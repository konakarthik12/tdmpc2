'''
This code is based on Transform2Act's parser.
https://github.com/Khrylx/Transform2Act/blob/main/khrylib/robot/xml_robot.py

We extend the baseline parser to accommodate environments in DMControl.
'''

import numpy as np
import math
from copy import deepcopy
from lxml.etree import XMLParser, parse
from lxml import etree
from io import BytesIO


def parse_vec(string):
    return np.fromstring(string, sep=' ')

def parse_fromto(string):
    fromto = np.fromstring(string, sep=' ')
    return fromto[:3], fromto[3:]

def normalize_range(value, lb, ub):
    return (value - lb) / (ub - lb) * 2 - 1

def denormalize_range(value, lb, ub):
    return (value + 1) * 0.5 * (ub - lb) + lb

def vec_to_polar(v):
    phi = math.atan2(v[1], v[0])
    theta = math.acos(v[2])
    return np.array([theta, phi])

def polar_to_vec(p):
    v = np.zeros(3)
    v[0] = math.sin(p[0]) * math.cos(p[1])
    v[1] = math.sin(p[0]) * math.sin(p[1])
    v[2] = math.cos(p[0])
    return v


class Joint:

    def __init__(self, node, node_attrib, body):
        self.node = node
        self.body = body
        self.node_attrib = deepcopy(node_attrib)
        
        self.name = node_attrib['name']
        self.type = node_attrib['type']
        
        self.range = np.deg2rad(parse_vec(node_attrib.get('range', "0 0")))
        self.pos = parse_vec(node_attrib.get('pos', '0 0 0'))
        self.axis = parse_vec(node_attrib.get('axis', '0 0 1'))
        
        # @TODO: actuator?
        '''
        actu_node = body.tree.getroot().find("actuator").find(f'motor[@joint="{self.name}"]')
        if actu_node is not None:
            self.actuator = Actuator(actu_node, self)
        else:
            self.actuator = None
        '''
        
    def __repr__(self):
        return 'joint_' + self.name

class Geom:

    def __init__(self, node, node_attrib, body):
        self.node = node
        self.body = body
        self.node_attrib = deepcopy(node_attrib)
        
        self.name = node_attrib.get('name', '')
        self.type = node_attrib['type']
        self.param_inited = False
        
        # attributes related to shape of this geom
        self.size = parse_vec(node_attrib['size'])
        self.pos = parse_vec(node_attrib.get('pos', '0 0 0'))
        self.fromto = parse_vec(node_attrib.get('fromto', '0 0 0 0 0 0'))
        
    def __repr__(self):
        return 'geom_' + self.name

class Actuator:

    def __init__(self, node, joint):
        self.node = node
        self.joint = joint
        self.cfg = joint.cfg
        self.joint_name = node.attrib['joint']
        self.name = self.joint_name
        self.parse_param_specs()
        self.param_inited = False
        # tunable parameters
        self.gear = float(node.attrib['gear'])

    def parse_param_specs(self):
        self.param_specs =  deepcopy(self.cfg['actuator_params'])
        for name, specs in self.param_specs.items():
            if 'lb' in specs and isinstance(specs['lb'], list):
                specs['lb'] = np.array(specs['lb'])
            if 'ub' in specs and isinstance(specs['ub'], list):
                specs['ub'] = np.array(specs['ub'])

    def sync_node(self):
        self.node.attrib['gear'] = f'{self.gear:.6f}'.rstrip('0').rstrip('.')
        self.name = self.joint.name
        self.node.attrib['name'] = self.name
        self.node.attrib['joint'] = self.joint.name

    def get_params(self, param_list, get_name=False):
        if 'gear' in self.param_specs:
            if get_name:
                param_list.append('gear')
            else:
                if not self.param_inited and self.param_specs['gear'].get('rel', False):
                    self.param_specs['gear']['lb'] += self.gear
                    self.param_specs['gear']['ub'] += self.gear
                    self.param_specs['gear']['lb'] = max(self.param_specs['gear']['lb'], self.param_specs['gear'].get('min', -np.inf))
                    self.param_specs['gear']['ub'] = min(self.param_specs['gear']['ub'], self.param_specs['gear'].get('max', np.inf))
                gear = normalize_range(self.gear, self.param_specs['gear']['lb'], self.param_specs['gear']['ub'])
                param_list.append(np.array([gear]))

        if not get_name:
            self.param_inited = True

    def set_params(self, params):
        if 'gear' in self.param_specs:
            self.gear = denormalize_range(params[0].item(), self.param_specs['gear']['lb'], self.param_specs['gear']['ub'])
            params = params[1:]
        return params


class Body:

    def __init__(self, node, parent_body, robot):
        self.node = node
        self.parent = parent_body
        if parent_body is not None:
            parent_body.child.append(self)
            parent_body.cind += 1
            self.depth = parent_body.depth + 1
            
            # find if any parent_body had 'childclass' tag
            child_class_parent_body = parent_body
            child_class = None
            while child_class_parent_body is not None:
                if child_class_parent_body.node.attrib.get('childclass', None) is not None:
                    child_class = child_class_parent_body.node.attrib.get('childclass', None)
                    break
                child_class_parent_body = child_class_parent_body.parent
        else:
            self.depth = 0
            child_class = node.attrib.get('childclass', None)
        self.robot = robot
        self.tree = robot.tree
        self.local_coord = robot.local_coord
        self.name = node.attrib['name'] if 'name' in node.attrib else self.parent.name + f'_child{len(self.parent.child)}'
        self.child = []
        self.cind = 0
        self.pos = parse_vec(node.attrib['pos'])
        
        # @sanghyun: DMControl specifies 'pos' in local coordinates
        # and we just keep it as it is, not changing it to global coordinates
        # if self.local_coord and parent_body is not None:
        #     self.pos += parent_body.pos
        
        # if cfg.get('init_root_from_geom', False):
        #     self.bone_start = None if parent_body is None else self.pos.copy()
        # else:
        #     self.bone_start = self.pos.copy()
        self.bone_start = self.pos.copy()
            
        print(f"=== Init body: {self.name} ===")
        
        '''
        Initialize joints
        '''
        # get default joint setting for 'default_root' or 'body' class
        # if there is 'body', it will be used.
        self.joints = []
        joint_entries = node.findall('joint')
        if 'default_root' in self.robot.defaults.keys():
            default_joint_setting = self.robot.defaults.get('default_root', {}).get('joint', {})
        if 'body' in self.robot.defaults.keys():
            default_joint_setting = self.robot.defaults.get('body', {}).get('joint', {})
        for je in joint_entries:
            joint_setting = deepcopy(default_joint_setting)
            
            # if there is a class attribute, use it to get default setting
            je_class = child_class
            if je.attrib.get('class', None) is not None:
                je_class = je.attrib.get('class')
                
            if je_class is not None and je_class in self.robot.defaults.keys():
                if 'joint' in self.robot.defaults[je_class].keys():
                    this_joint_setting = self.robot.defaults[je_class].get('joint')
                    joint_setting.update(this_joint_setting)
                    
            # update joint setting with the attributes of the joint element
            # this becomes the final setting for the joint
            joint_setting.update(je.attrib)
            
            # print(f"\t=== orig  joint attrib: {je.attrib}")
            # print(f"\t=== final joint attrib (applied default settings): {joint_setting}")
            # print()
                    
            joint = Joint(je, joint_setting, self)
            self.joints.append(joint)
            
        '''
        Initialize geoms
        '''
        # get default geom setting for 'default_root' or 'body' class
        # if there is 'body', it will be used.
        self.geoms = []
        geom_entries = node.findall('geom')
        if 'default_root' in self.robot.defaults.keys():
            default_geom_setting = self.robot.defaults.get('default_root', {}).get('geom', {})
        if 'body' in self.robot.defaults.keys():
            default_geom_setting = self.robot.defaults.get('body', {}).get('geom', {})
        for ge in geom_entries:
            geom_setting = deepcopy(default_geom_setting)
            
            # if there is a class attribute, use it to get default setting
            ge_class = child_class
            if ge.attrib.get('class', None) is not None:
                ge_class = ge.attrib.get('class')
                
            if ge_class is not None:
                if ge_class in self.robot.defaults.keys():
                    if 'geom' in self.robot.defaults[ge_class].keys():
                        this_geom_setting = self.robot.defaults[ge_class].get('geom')
                        geom_setting.update(this_geom_setting)
            
            # update geom setting with the attributes of the geom element
            # this becomes the final setting for the geom
            geom_setting.update(ge.attrib)
            
            # print(f"\t=== orig  geom attrib: {ge.attrib}")
            # print(f"\t=== final geom attrib (applied default settings): {geom_setting}")
            # print()
            
            geom = Geom(ge, geom_setting, self)
            self.geoms.append(geom)
            
        self.param_inited = False
        # parameters
        self.bone_end = None
        self.bone_offset = None

    def __repr__(self):
        return 'body_' + self.name

class Robot:

    def __init__(self, xml):
        self.bodies = []
        self.tree = None        # xml tree
        self.defaults = {}      # [class_name][attr_name] = attr_value
        self.load_from_xml(xml)
        
    def load_from_xml(self, xml):
        self.tree = parse(BytesIO(xml))
        
        # not all models have compiler tag
        compiler = self.tree.getroot().find('.//compiler')
        if compiler is not None:
            self.local_coord = compiler.attrib['coordinate'] == 'local'
        else:
            self.local_coord = False
            
        # default settings
        root = self.tree.getroot().find('default')
        self.set_defaults(root, None)
            
        root = self.tree.getroot().find('worldbody').find('body')
        self.add_body(root, None)
        
    def set_defaults(self, root, parent_class):
        
        if parent_class is not None:
            class_name = root.attrib['class']
            self.defaults[class_name] = deepcopy(self.defaults[parent_class])
        else:
            class_name = 'default_root'
            
            # default settings of Mujoco
            self.defaults[class_name] = {
                'joint': {
                    # https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint
                    'type': 'hinge',
                    'pos': '0 0 0',
                    'axis': '0 0 1',
                    'range': '0 0',
                },
                'geom': {
                    # https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-geom
                    'type': "sphere",
                    'size': '0 0 0',
                    'pos': '0 0 0',
                    'fromto': '0 0 0 0 0 0',
                }
            }
        
        default_class_children = []
        children = root.getchildren()
        for child in children:
            # if this child defines a new default class
            if child.tag == 'default':
                default_class_children.append(child)
            else:
                if child.tag in self.defaults[class_name].keys():
                    self.defaults[class_name][child.tag].update(child.attrib)
                else:
                    self.defaults[class_name][child.tag] = child.attrib
            
        for child in default_class_children:
            self.set_defaults(child, class_name)
            
    def add_body(self, body_node, parent_body):
        body = Body(body_node, parent_body, self)
        self.bodies.append(body)

        for body_node_c in body_node.findall('body'):
            self.add_body(body_node_c, body)

    def write_xml(self, fname):
        self.tree.write(fname, pretty_print=True)

    def export_xml_string(self):
        return etree.tostring(self.tree, pretty_print=True)

    def get_gnn_edges(self):
        edges = []
        for i, body in enumerate(self.bodies):
            if body.parent is not None:
                j = self.bodies.index(body.parent)
                edges.append([i, j])
                edges.append([j, i])
        edges = np.stack(edges, axis=1)
        return edges