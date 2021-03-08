from re import T
import sympy as sp
from sympy.physics.vector import ReferenceFrame, Point, Vector
from sympy.printing.defaults import Printable
from sympy.vector import BodyOrienter
from sympy.vector.orienters import Orienter
import numpy as np


class TF(Printable):
    def __init__(self, name,
                 position=None,
                 rotation=None,
                 rotation_type=None,
                 rotation_order=None,
                 parent=None):

        rotation_order = rotation_order or 'zyx'
        rotation_type = rotation_type or 'Body'

        rf_name = f'{name}RF'
        origo_name = f'{name}O'

        self.pos_default = sp.Matrix(sp.symbols(f'{name}_x:z'))
        self.rot_default = sp.Matrix(sp.symbols(f'{name}_p:r'))

        # self.pos = position or self.pos_default.copy()
        # self.rot = rotation or self.rot_default.copy()

        # self.rot_type = rotation_type
        self.rot_order = rotation_order
        self.name = name

        if parent is None:
            self.parent = self
            self.rf = ReferenceFrame(rf_name)
            self.origo = Point(origo_name)

        else:
            self.parent = parent
            rot = rotation
            rot = sp.Matrix(rot) if rot is not None else self.rot_default
            if rotation_type.lower() in ['body', 'axis']:
                rot = self.rot_reorder(rot, rotation_order)

            self.rf = parent.rf.orientnew(
                rf_name, rotation_type,
                sp.Matrix(rot),
                rotation_order)

            self.origo = Point(origo_name)
            pos = position
            self.set_position(
                sp.Matrix(pos) if pos is not None else self.pos_default)

    @property
    def free_symbols(self):
        parent = self.parent.free_symbols if self.parent is not self else []
        origo = self.origo.pos_from(
            self.parent.origo).free_symbols(self.parent.rf)
        rf = self.rf.dcm(self.parent.rf).free_symbols

        return list(parent + list(origo) + list(rf))

    @ property
    def t_mat_parent(self):
        rot = self.parent.rf.dcm(self.rf)
        transform = self.origo.pos_from(
            self.parent.origo).to_matrix(self.parent.rf)
        return sp.Matrix([[rot, transform],
                          [0, 0, 0, 1]])

    @ property
    def t_mat(self):
        if self.parent is self:
            return self.t_mat_parent
        else:
            return self.parent.t_mat * self.t_mat_parent

    def set_position(self, position):
        position = [i if i is not None else j
                    for i, j in zip(position or 3 * [None], self.pos_default)]

        self.origo.set_pos(self.parent.origo,
                           Vector([(sp.Matrix(position), self.parent.rf)]))

    def set_orientation(self, orienter):
        """
        For some reason orienter rotation matrix has to be inverted
        """
        self.orientation = orienter

        def not_parent_filter(key_and_item):
            return key_and_item[0] is not self.parent.rf

        dcm_dict_old = self.rf._dcm_dict.copy()
        dcm_children = dict()
        for key, item in filter(not_parent_filter, dcm_dict_old.items()):
            dcm_children[key] = key._dcm_dict[self.rf]

        self.rf.orient(self.parent.rf, 'DCM', orienter.rotation_matrix().inv())

        for key, item in filter(not_parent_filter, dcm_dict_old.items()):
            self.rf._dcm_dict[key] = item
            key._dcm_dict[self.rf] = dcm_children[key]

    def set_orientation_body(self, rotations, rotation_order=None):
        rotations = [i if i is not None else j
                     for i, j in zip(rotations or 3 * [None], self.rot_default)]
        self.rot_order = rotation_order or self.rot_order

        orienter = BodyOrienter(
            *self.rot_reorder(rotations, self.rot_order), self.rot_order)
        self.set_orientation(orienter)

    def new_TF(self, name, position=None, rotation=None, rotation_type=None,
               rotation_order=None):
        return TF(name=name, position=position, rotation=rotation,
                  rotation_type=rotation_type, rotation_order=rotation_order,
                  parent=self)

    def new_TF_from_t_mat(self, name, t_mat):
        position = sp.Matrix(t_mat[:3, 3])
        rotation = sp.Matrix(t_mat[:3, :3])
        return TF(name=name, position=position, rotation=rotation,
                  rotation_type='DCM', parent=self)

    def new_point(self, name, position):
        point = self.origo.locatenew(
            name, Vector([(sp.Matrix(position), self.rf)]))
        return point

    def new_points(self, names, positions):
        if isinstance(names, str):
            names = [names + f'_{i}' for i in range(len(positions))]
        return [self.new_point(name, pos)
                for name, pos in zip(names, positions)]

    def project_point(self, point):
        return point.pos_from(self.origo).to_matrix(self.rf)

    def project_points(self, points):
        return sp.Matrix([[self.project_point(point) for point in points]])

    def _repr_latex_(self): return f"$\\displaystyle {self.name}$"

    def __str__(self): return self.name

    @staticmethod
    def rot_reorder(pqr, order):
        return sp.Matrix([pqr[ord(c) - ord('x')] for c in order])
