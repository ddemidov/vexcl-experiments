#include <vexcl/vexcl.hpp>

/*
 * Variant 1: using enable_if.
 * The drawback is that unspecialized template has to know about its
 * specializations.
 */
template <class T>
typename std::enable_if<
    !vex::is_vector_expression<T>::value,
    void
>::type
process_v1(int i, T&&) {
    std::cout << i << ": unspecialized" << std::endl;
}

template <class T>
typename std::enable_if<
    vex::is_vector_expression<T>::value,
    void
>::type
process_v1(int i, T &&t) {
    std::cout << i << ": VexCL expression: ";
    boost::proto::display_expr(t, std::cout);
}

/*
 * Variant 2: using tags.
 */
struct general_tag {};
struct vexcl_expr_tag {};

template <class T, class Enable = void>
struct tag_of {
    typedef general_tag type;
};

template <class T>
struct tag_of<T, typename std::enable_if< vex::is_vector_expression<T>::value>::type>
{
    typedef vexcl_expr_tag type;
};

template <class T>
void process_v2_dispatch(int i, T&&, general_tag) {
    std::cout << i << ": unspecialized" << std::endl;
}

template <class T>
void process_v2_dispatch(int i, T &&t, vexcl_expr_tag) {
    std::cout << i << ": VexCL expression: ";
    boost::proto::display_expr(t, std::cout);
}

template <class T>
void process_v2(int i, T &&t) {
    typedef typename tag_of< typename std::decay<T>::type >::type tag;

    process_v2_dispatch(i, std::forward<T>(t), tag());
}

int main() {
    vex::Context ctx(vex::Filter::Env);

    vex::vector<int> x(ctx, 1024);
    vex::vector<int> y(ctx, 1024);

    std::cout << "Version 1:" << std::endl;
    process_v1(1, 42);
    process_v1(2, x);
    process_v1(3, x + y);

    std::cout << std::endl << "Version 2:" << std::endl;
    process_v2(1, 42);
    process_v2(2, x);
    process_v2(3, x + y);

}
