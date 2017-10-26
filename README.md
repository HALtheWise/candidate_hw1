# Eric Miller - Overhang Detection

![](problem-1/cube.gif)

## Observations

This problem sounds easy at first glance, but a number of edge cases make it tricky.
1. The "house" shape, which should need no support at all because the sloped roof holds up the top edge.
2. Anything touching the build plate, which doesn't need support.
3. A Cube, where the top edges don't need support, but the top face does.
4. A stalagtite, where the tip vertex needs support, but none of the edges or faces do.

## Plan

The basic approach I want to take is to determine whether a face/edge/vertex needs support purely by looking at the local properties of that feature and the things around it. We'll see if that works.

An alternative approach would be to build a directed graph indicating what things hold each other up, and reason about which edges need to be added to make it have a root.

An STL file contains information about faces, edges, and vertices.
### Faces
A face needs support when its is "flat" i.e. its normal vectors is close to the gravity vector. A cross product to get the normal vector, followed by a dot product with the gravity vector, makes computing that really easy.

Precomputation of $cos(\theta)$ should make the whole computation quite cheap.

### Edges
In addition to edges surrounding faces needing support, some edges themselves need to be supported, even if both of their neighboring faces are fine. An example is a horizontal edge connecting two faces in a "V" configuration. Notice that this edge requires support, while the inverse case (a peak) doesn't.

In this case, we are looking for an edge that is mostly horizontal (dot product again), and where neither (none) of its neighboring faces is sufficiently below it to support it (using their normals). Depending on what's next in the pipeline, it may or may not matter whether an edge adjacent to a supported face is supported or not.

This assumes that your FDM machine cannot perform bridging operations.

### Vertices

In addition to the cases above, a vertex needs support if none of its neighboring vertices are sufficiently below it. (i.e. tip of a [stalagtite](http://media.gettyimages.com/photos/stalactites-and-stalagmites-in-jenolan-caves-picture-id595906719?s=612x612) )


![](problem-1/part.gif)

# --Original assignment below--
This readme contains problems that candidates can choose to work on as a take-home assignment

### Instructions:
1. Fork this repo
2. Choose at least one of the problems from this repo to solve
    * You can use any language you want but python is recommended
    * Depending on your time and level of enthusiasm about the problem you can choose to do only parts of the problem
3. Once done send the link to your repo back to your interviewer

Open an issue on this repo if you have any questions about the problems.

Adding clarification and description as comments or readme file is welcomed if needed.

### Problem 1: Path Planning
You are working on a path planning program for a 3D printer that prints using the FDM process. If you are not familiar with the FDM process read [its Wikipedia page](https://en.wikipedia.org/wiki/Fused_deposition_modeling). Given an STL file, a vector that indicates gravity, and maximum overhang angle that the printer can support, the goal is to find all vertices and edges from the STL file that require support to be printed. An STL file (part.STL) is included in this repo that can be used for testing and demonstrating your application.

1. STL file, maximum overhang angle, and gravity vector are inputs to your program
2. Assume build plate will be normal to gravity vector
3. Assume that your STL is always a shell structure (thin-walled)
3. To show your results, you can show all vertices and edges from STL file and differently color the ones that require support

### Problem 2 : Video Analysis
You are a control engineer for a robotic welder. You have a camera that records the welding process. You want to develop a program that detects certain features and events in your process which then allows you to trigger other controls required for the process. A sample video of a weld process is included in this repo (sample_weld_video.mp4) that can be used to test and demonstrate your program.

Your program is required to do following:
1. Identify the location of the weld pool in all frames
2. Detect dropped frames/skipping events
3. Detect welder on/off events
4. Detect motion stopping

Optional: If you feel like a super star you can work on any of the following as well:
1. Detect spatter
2. Inclination of the weld path (in frame)
3. Length of the wire between tip and weld pool
4. Angle between the wire and weld pool
5. Count pixels that can be used to estimate height of weld bead

