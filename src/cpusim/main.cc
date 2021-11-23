#include <iostream>

#include "macros.h"
#include "serde.h"

int main(int argc, char *argv[])
{
    CORE::XYZ v{3, 4, 5};
    CORE::XYZ u{30, 40, 50};
    CORE::XYZ w{40, 50, 60};
    v += u;
    w = v + u;
    v -= u;
    w = v - u;
    w = -u;
    v *= 2.0f;
    w = v * 2.0f;
    w = 2.0f * v;
    v /= 2.0f;
    w = v / 2.0f;
    std::cout << "u=" << u << std::endl;
    std::cout << "v=" << v << std::endl;
    std::cout << "w=" << w << std::endl;
    std::cout << "This is very good" << std::endl;

    std::cout << argc << std::endl;
    for (int i = 0; i < argc; i++)
    {
        std::cout << argv[i] << std::endl;
    }

    return 0;
}