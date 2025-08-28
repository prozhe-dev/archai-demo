export type Point = [number, number]; // A single coordinate [x, y]

export type Line = [Point, Point]; // A line segment between two points

export type Shape = Line[]; // A shape is composed of multiple line segments

export type Triplet = [number, number, number];
