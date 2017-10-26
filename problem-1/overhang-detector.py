# coding: utf-8

import numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits import mplot3d
# Using library numpy-stl to handle STL files (https://pypi.python.org/pypi/numpy-stl)
from stl.mesh import Mesh


# # Observations
# 
# This problem sounds easy at first glance, but a number of edge cases make it tricky.
# 1. The "house" shape, which should need no support at all because the sloped roof holds up the top edge.
# 2. Anything touching the build plate, which doesn't need support.
# 3. A Cube, where the top edges don't need support, but the top face does.
# 4. A stalactite, where the tip vertex needs support, but none of the edges or faces do.

# # Plan

# The basic approach I want to take is to determine whether a face/edge/vertex needs support purely by
# looking at the local properties of that feature and the things around it. We'll see if that works.
# 
# An alternative approach would be to build a directed graph indicating what things hold each other up, and
# reason about which edges need to be added to make it have a root.

# An STL file contains information about faces, edges, and vertices.
# ## Faces
# A face needs support when its is "flat" i.e. its normal vectors is close to the gravity vector. A cross product
# to get the normal vector, followed by a dot product with the gravity vector, makes computing that really easy.
# 
# Precomputation of $cos(\theta)$ should make the whole computation quite cheap.
# 
# ## Edges
# In addition to edges surrounding faces needing support, some edges themselves need to be supported, even if both
# of their neighboring faces are fine. An example is a horizontal edge connecting two faces in a "V" configuration.
# Notice that this edge requires support, while the inverse case (a peak) doesn't.
# 
# In this case, we are looking for an edge that is mostly horizontal (dot product again), and where neither (none)
# of its neighboring faces is sufficiently below it to support it (using their normals). Depending on what's next
# in the pipeline, it may or may not matter whether an edge adjacent to a supported face is supported or not.
# 
# This assumes that your FDM machine cannot perform bridging operations.
# 
# ## Vertices
# 
# In addition to the cases above, a vertex needs support if none of its neighboring vertices are sufficiently below it.
# (i.e. tip of a stalactite)

class OverhangDetector(object):
    epsilon = 0.000001  # A very small number, to deal with floating point rounding.

    def __init__(self, overhang=np.deg2rad(45), down=(0, 0, -1)):
        self.overhang = overhang
        self.down = np.array(down)

        self.costheta = np.cos(self.overhang)

    def isMostlyUp(self, vec):
        """
        Returns true if the provided 3-vector is pointning within theta of up
        """
        up = -self.down
        cosangle = np.dot(vec, up) / (np.linalg.norm(up) * np.linalg.norm(vec))

        return cosangle >= self.costheta

    def isMostlyVertical(self, vec):
        """
        Returns true if the provided 3-vector is pointning within theta of up or down
        """
        cosangle = np.dot(vec, self.down) / (np.linalg.norm(self.down) * np.linalg.norm(vec))

        return cosangle >= self.costheta or cosangle <= -self.costheta

    def almostEqual(self, v1, v2):
        """
        Checks whether two values (numbers, vectors, or arrays) are equal within epsilon.
        """
        return np.max(np.abs(v2 - v1)) <= self.epsilon

    def isOnBuildPlate(self, point):
        # The build plate is assumed to be a infinite plane perpendicular
        # to gravity, passing through the point (0, 0, 0)
        return self.almostEqual(np.dot(point, self.down), 0)

    def faceNeedsSupport(self, face):
        """
        face is a list of three vertices
        """
        p1, p2, p3 = np.array(face)
        normal = np.cross(p2 - p1, p3 - p1)

        if not self.isMostlyVertical(normal):
            # The face itself is vertical
            return False
        else:
            # The face is horizontal
            for p in face:
                if not self.isOnBuildPlate(p):
                    # At least one vertex is not on the build plate
                    return True
            # All vertices on build plate
            return False

    def edgeNeedsSupport(self, endpoints, faces):
        """endpoints is list of two points,
        faces is a list of arbitrarily many neighboring faces, given as triples of points"""

        p1, p2 = np.array(endpoints)

        if self.isMostlyVertical(p2 - p1):
            # Vertical edges are supported by their endpoint
            return False

        if self.isOnBuildPlate(p1) and self.isOnBuildPlate(p2):
            # This edge is supported by the build plate
            return False

        # Check for a face that supports this edge
        for face in faces:
            # We need to check whether the face is "below" the edge.
            # Note that this can happen even if the third point on the edge is above
            # both of the points on the edge.

            otherPoints = [p for p in face if (not self.almostEqual(p, p1) and not self.almostEqual(p, p2))]
            if len(otherPoints) != 1:
                raise Exception('Face provided not containing given edge')

            p3 = np.array(otherPoints[0])

            # The easiest way to check this is to which side of this edge the face is on
            # is to compute the cross product of this edge's vector with the vector to the third point,
            # as well as the cross product of this edge's vector and gravity. If those vectors are within
            # +-90 degrees of each other, the face is below the edge.

            faceNormal = np.cross((p2 - p1), (p3 - p1))
            gravityNormal = np.cross((p2 - p1), self.down)

            #         faceNormal /= np.linalg.norm(faceNormal)
            #         gravityNormal /= np.linalg.norm(gravityNormal)

            if np.dot(faceNormal, gravityNormal) > 0:
                # The face is "below" this edge
                if not self.faceNeedsSupport(face):
                    # The face can support this edge
                    return False

        return True

    def vertexNeedsSupport(self, point, edges):
        """
        Takes as input:
            - A point Vector3
            - A list of edges connected to this point, given as pairs of points
        """
        point = np.array(point)
        edges = np.array(edges)

        if self.isOnBuildPlate(point):
            return False

        for edge in edges:
            otherPoints = [p for p in edge if not self.almostEqual(p, point)]
            if len(otherPoints) != 1:
                raise Exception('Edge provided not containing given vertex')
            other = otherPoints[0]

            # Check whether this point is mostly above the current edge
            if self.isMostlyUp(point - other):
                # This vertex is at the top of the provided edge, and so supported.
                return False

        return True

    def splitFaces(self, faces):
        """

        :return: tuple of (faces, edges, vertices)
        """
        vertices = defaultdict(
            set)  # Map of vertex (tuple) to set of neighboring edges, each represented as a sorted tuple
        edges = defaultdict(
            set)  # Map of edge (sorted tuple) to set of neighboring faces, each represented as a sorted tuple

        # faces = set() # set of faces (sorted tuples)

        # TODO: Refactor to not use nested functions
        def add_vertex(vertex, edge):
            edge = tuple(sorted(edge))
            if vertex not in edge:
                raise Exception('Vertex must be in edge')
            vertices[vertex].add(edge)

        def add_edge(edge, face):
            edge = tuple(sorted(edge))
            face = tuple(sorted(face))

            p1, p2 = edge
            add_vertex(p1, edge)
            add_vertex(p2, edge)

            edges[edge].add(face)

        def add_face(face):
            face = tuple(sorted(face))

            p1, p2, p3 = face
            add_edge((p1, p2), face)
            add_edge((p2, p3), face)
            add_edge((p1, p3), face)

        for f in faces:
            add_face(map(tuple, f))

        return faces, edges, vertices

    # Create a new plot
    def solveAndDisp(self, faces, showFaces=True, showEdges=True, showVertices=True):
        faces, edges, vertices = self.splitFaces(faces)

        fig = plt.figure()
        axes = mplot3d.Axes3D(fig)

        axes.set_xlabel('x')
        axes.set_ylabel('y')
        axes.set_zlabel('z')

        # TODO: Automatically determine appropriate bounds based on model data
        axes.set_xlim(-20, 20)
        axes.set_ylim(-20, 20)
        axes.set_zlim(0, 40)

        red = (1, 0, 0)
        blue = (0, 0, 1)

        if showEdges:
            edgeskeys = edges.keys()

            linecollection = mplot3d.art3d.Line3DCollection(edgeskeys, alpha=1)
            linecollection.set_color([red if self.edgeNeedsSupport(e, edges[e]) else blue for e in edgeskeys])

            axes.add_collection3d(linecollection)

        if showFaces:
            facecollection = mplot3d.art3d.Poly3DCollection(faces, alpha=0.1)
            facecollection.set_color([red if self.faceNeedsSupport(f) else blue for f in faces])

            axes.add_collection3d(facecollection)

        if showVertices:
            for p, es in vertices.iteritems():
                circle = Circle(p[0:2], 1)
                circle.set_color(red if self.vertexNeedsSupport(p, list(es)) else blue)
                axes.add_patch(circle)
                mplot3d.art3d.pathpatch_2d_to_3d(circle, z=p[2])

        return fig


if __name__ == '__main__':
    detector = OverhangDetector()

    detector.solveAndDisp(Mesh.from_file('tests/octahedron.stl').vectors).show()
    detector.solveAndDisp(Mesh.from_file('tests/cube.stl').vectors).show()

    detector.solveAndDisp(Mesh.from_file('part.stl').vectors, showVertices=False).show()

    while True:
        pass
