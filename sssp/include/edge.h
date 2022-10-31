#pragma once

typedef unsigned int uint;
typedef unsigned long long ull;

struct OutEdge
{
    uint end;
};

struct OutEdgeWeighted
{
    uint end;
    uint w8;
};

struct Edge
{
    uint source;
    uint end;
};

struct EdgeWeighted
{
    uint source;
    uint end;
    uint w8;
};
