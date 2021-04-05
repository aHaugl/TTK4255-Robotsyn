import sympy as sp
from sympy.physics.vector import ReferenceFrame, Point, Vector
from sympy.printing.defaults import Printable
from sympy.vector import BodyOrienter


class TF(Printable):
    """
    Class representing a transform. It is basically a combination of the 
    Point adn ReferenceFrame from sympy.physics.vector

    This class aims to make it easy to get symbolc transformation matrices from
    one frame to another. 
    """

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

        self.rot_type = rotation_type
        self.rot_order = rotation_order
        self.name = name

        if parent is None:
            self.parent = self
            self.rf = ReferenceFrame(rf_name)
            self.origo = Point(origo_name)

        else:
            self.parent = parent
            if rotation is None:
                rotation = self.rot_default
            elif rotation_type.lower() in ['body', 'axis']:
                rotation = [i if i is not None else j
                            for i, j in zip(rotation, self.rot_default)]

            rot = sp.Matrix(rotation)
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

        return list(parent
                    + sorted(list(origo), key=str)
                    + sorted(list(rf), key=str))

    def t_mat(self, otherframe):
        """
        Transform matrix from otherframe to self
        """
        rot = self.rf.dcm(otherframe.rf)
        translation = self.project_point(otherframe.origo)
        return sp.Matrix([[rot, translation],
                          [0, 0, 0, 1]])

    def set_position(self, position):
        """ Set position wrt parent"""
        position = [i if i is not None else j
                    for i, j in zip(position or 3 * [None], self.pos_default)]

        self.origo.set_pos(self.parent.origo,
                           Vector([(sp.Matrix(position), self.parent.rf)]))

    def set_orientation(self, orienter):
        """
        This func rotates the TF wrt parent and keeps the rot with children
        For some reason orienter rotation matrix has to be inverted???
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
        """ in sympy the inputs are the order of the rotations, we want pqr
        """
        return sp.Matrix([pqr[ord(c) - ord('x')] for c in order])
