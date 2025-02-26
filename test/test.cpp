#include <iostream>
#include <memory>
#include <vector>

class A {
public:
    A(double x) : x(x) {}
    double x;
    virtual void print() {
        std::cout << "class A" << std::endl;
    }

    virtual A* clone() const {
        return new A(x);
    }

    template<typename T, typename P>
    friend std::unique_ptr<T> operator+(const T& a, const P& b)
    {
        auto result = a.clone();
        result->x = a.x + b.x;
        return std::unique_ptr<T>(result);
    }
};

class B : public A {
public:
    B(double x) : A(x) {}
    virtual void print() override {
        std::cout << "class B" << std::endl;
    }

    B* clone() const override {
        return new B(x);
    }
};

int main() {
    std::vector<std::shared_ptr<A>> vec;
    vec.push_back(std::make_shared<B>(2.0));
    for (auto& a : vec) {
        auto c = a;
        c->print();
        (*c).print();
        (*c + *c)->print();
    }
    return 0;
}
