#include <Python.h>
#include "pytensor_mod_helper.h"
#include <math.h>
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
#include <numpy/npy_math.h>
#include <vector>
#include <algorithm>
//////////////////////
////  Support Code
//////////////////////

// For GPU support
            #ifdef WITHIN_KERNEL
            #define DEVICE WITHIN_KERNEL
            #else
            #define DEVICE
            #endif

            #ifndef M_PI
            #define M_PI 3.14159265358979323846
            #endif

            #ifndef _PSIFUNCDEFINED
            #define _PSIFUNCDEFINED
            DEVICE double _psi(double x) {

                /*taken from
                Bernardo, J. M. (1976). Algorithm AS 103:
                Psi (Digamma) Function. Applied Statistics. 25 (3), 315-317.
                http://www.uv.es/~bernardo/1976AppStatist.pdf
                */

                double y, R, psi_ = 0;
                double S  = 1.0e-5;
                double C = 8.5;
                double S3 = 8.333333333e-2;
                double S4 = 8.333333333e-3;
                double S5 = 3.968253968e-3;
                double D1 = -0.5772156649;

                if (x <= 0) {
                    // the digamma function approaches infinity from one side and -infinity from the other, around negative integers and zero
                    if (x == floor(x)) {
                        return INFINITY; // note that scipy returns -INF for 0 and NaN for negative integers
                    }

                    // Use reflection formula
                    double pi_x = M_PI * x;
                    double cot_pi_x = cos(pi_x) / sin(pi_x);
                    return _psi(1.0 - x) - M_PI * cot_pi_x;
                }

                y = x;

                if (y <= S)
                    return D1 - 1.0/y;

                while (y < C) {
                    psi_ = psi_ - 1.0 / y;
                    y = y + 1;
                }

                R = 1.0 / y;
                psi_ = psi_ + log(y) - .5 * R ;
                R= R*R;
                psi_ = psi_ - R * (S3 - R * (S4 - R * S5));

                return psi_;
            }
            #endif

    namespace {
    struct __struct_compiled_op_mbd9a99ccae6f8313aa7113832d471284ef2a555e009e6ae103d62f0b8be82a12 {
        PyObject* __ERROR;

        PyObject* storage_V51;
PyObject* storage_V49;
PyObject* storage_V47;
PyObject* storage_V45;
PyObject* storage_V43;
PyObject* storage_V41;
PyObject* storage_V39;
PyObject* storage_V37;
PyObject* storage_V35;
PyObject* storage_V33;
PyObject* storage_V31;
PyObject* storage_V29;
PyObject* storage_V27;
PyObject* storage_V25;
PyObject* storage_V23;
PyObject* storage_V21;
PyObject* storage_V1;
PyObject* storage_V3;
PyObject* storage_V5;
PyObject* storage_V7;
PyObject* storage_V9;
PyObject* storage_V11;
PyObject* storage_V13;
PyObject* storage_V15;
PyObject* storage_V17;
PyObject* storage_V19;
        

        __struct_compiled_op_mbd9a99ccae6f8313aa7113832d471284ef2a555e009e6ae103d62f0b8be82a12() {
            // This is only somewhat safe because we:
            //  1) Are not a virtual class
            //  2) Do not use any virtual classes in the members
            //  3) Deal with mostly POD and pointers

            // If this changes, we would have to revise this, but for
            // now I am tired of chasing segfaults because
            // initialization code had an error and some pointer has
            // a junk value.
            #ifndef PYTENSOR_DONT_MEMSET_STRUCT
            memset(this, 0, sizeof(*this));
            #endif
        }
        ~__struct_compiled_op_mbd9a99ccae6f8313aa7113832d471284ef2a555e009e6ae103d62f0b8be82a12(void) {
            cleanup();
        }

        int init(PyObject* __ERROR, PyObject* storage_V51, PyObject* storage_V49, PyObject* storage_V47, PyObject* storage_V45, PyObject* storage_V43, PyObject* storage_V41, PyObject* storage_V39, PyObject* storage_V37, PyObject* storage_V35, PyObject* storage_V33, PyObject* storage_V31, PyObject* storage_V29, PyObject* storage_V27, PyObject* storage_V25, PyObject* storage_V23, PyObject* storage_V21, PyObject* storage_V1, PyObject* storage_V3, PyObject* storage_V5, PyObject* storage_V7, PyObject* storage_V9, PyObject* storage_V11, PyObject* storage_V13, PyObject* storage_V15, PyObject* storage_V17, PyObject* storage_V19) {
            Py_XINCREF(storage_V51);
Py_XINCREF(storage_V49);
Py_XINCREF(storage_V47);
Py_XINCREF(storage_V45);
Py_XINCREF(storage_V43);
Py_XINCREF(storage_V41);
Py_XINCREF(storage_V39);
Py_XINCREF(storage_V37);
Py_XINCREF(storage_V35);
Py_XINCREF(storage_V33);
Py_XINCREF(storage_V31);
Py_XINCREF(storage_V29);
Py_XINCREF(storage_V27);
Py_XINCREF(storage_V25);
Py_XINCREF(storage_V23);
Py_XINCREF(storage_V21);
Py_XINCREF(storage_V1);
Py_XINCREF(storage_V3);
Py_XINCREF(storage_V5);
Py_XINCREF(storage_V7);
Py_XINCREF(storage_V9);
Py_XINCREF(storage_V11);
Py_XINCREF(storage_V13);
Py_XINCREF(storage_V15);
Py_XINCREF(storage_V17);
Py_XINCREF(storage_V19);
            this->storage_V51 = storage_V51;
this->storage_V49 = storage_V49;
this->storage_V47 = storage_V47;
this->storage_V45 = storage_V45;
this->storage_V43 = storage_V43;
this->storage_V41 = storage_V41;
this->storage_V39 = storage_V39;
this->storage_V37 = storage_V37;
this->storage_V35 = storage_V35;
this->storage_V33 = storage_V33;
this->storage_V31 = storage_V31;
this->storage_V29 = storage_V29;
this->storage_V27 = storage_V27;
this->storage_V25 = storage_V25;
this->storage_V23 = storage_V23;
this->storage_V21 = storage_V21;
this->storage_V1 = storage_V1;
this->storage_V3 = storage_V3;
this->storage_V5 = storage_V5;
this->storage_V7 = storage_V7;
this->storage_V9 = storage_V9;
this->storage_V11 = storage_V11;
this->storage_V13 = storage_V13;
this->storage_V15 = storage_V15;
this->storage_V17 = storage_V17;
this->storage_V19 = storage_V19;
            



























            this->__ERROR = __ERROR;
            return 0;
        }
        void cleanup(void) {
            __label_1:

double __DUMMY_1;
__label_3:

double __DUMMY_3;
__label_5:

double __DUMMY_5;
__label_7:

double __DUMMY_7;
__label_9:

double __DUMMY_9;
__label_11:

double __DUMMY_11;
__label_13:

double __DUMMY_13;
__label_15:

double __DUMMY_15;
__label_17:

double __DUMMY_17;
__label_19:

double __DUMMY_19;
__label_21:

double __DUMMY_21;
__label_23:

double __DUMMY_23;
__label_25:

double __DUMMY_25;
__label_27:

double __DUMMY_27;
__label_29:

double __DUMMY_29;
__label_31:

double __DUMMY_31;
__label_33:

double __DUMMY_33;
__label_35:

double __DUMMY_35;
__label_37:

double __DUMMY_37;
__label_39:

double __DUMMY_39;
__label_41:

double __DUMMY_41;
__label_43:

double __DUMMY_43;
__label_45:

double __DUMMY_45;
__label_47:

double __DUMMY_47;
__label_49:

double __DUMMY_49;
__label_51:

double __DUMMY_51;
__label_54:

double __DUMMY_54;

            Py_XDECREF(this->storage_V51);
Py_XDECREF(this->storage_V49);
Py_XDECREF(this->storage_V47);
Py_XDECREF(this->storage_V45);
Py_XDECREF(this->storage_V43);
Py_XDECREF(this->storage_V41);
Py_XDECREF(this->storage_V39);
Py_XDECREF(this->storage_V37);
Py_XDECREF(this->storage_V35);
Py_XDECREF(this->storage_V33);
Py_XDECREF(this->storage_V31);
Py_XDECREF(this->storage_V29);
Py_XDECREF(this->storage_V27);
Py_XDECREF(this->storage_V25);
Py_XDECREF(this->storage_V23);
Py_XDECREF(this->storage_V21);
Py_XDECREF(this->storage_V1);
Py_XDECREF(this->storage_V3);
Py_XDECREF(this->storage_V5);
Py_XDECREF(this->storage_V7);
Py_XDECREF(this->storage_V9);
Py_XDECREF(this->storage_V11);
Py_XDECREF(this->storage_V13);
Py_XDECREF(this->storage_V15);
Py_XDECREF(this->storage_V17);
Py_XDECREF(this->storage_V19);
        }
        int run(void) {
            int __failure = 0;
            
    PyObject* py_V1;
    
        PyArrayObject* V1;
        
            typedef npy_bool dtype_V1;
            
    PyObject* py_V3;
    
        PyArrayObject* V3;
        
            typedef npy_bool dtype_V3;
            
    PyObject* py_V5;
    
        PyArrayObject* V5;
        
            typedef npy_float64 dtype_V5;
            
    PyObject* py_V7;
    
        PyArrayObject* V7;
        
            typedef npy_float64 dtype_V7;
            
    PyObject* py_V9;
    
        PyArrayObject* V9;
        
            typedef npy_float64 dtype_V9;
            
    PyObject* py_V11;
    
        PyArrayObject* V11;
        
            typedef npy_float64 dtype_V11;
            
    PyObject* py_V13;
    
        PyArrayObject* V13;
        
            typedef npy_float64 dtype_V13;
            
    PyObject* py_V15;
    
        PyArrayObject* V15;
        
            typedef npy_float64 dtype_V15;
            
    PyObject* py_V17;
    
        PyArrayObject* V17;
        
            typedef npy_float64 dtype_V17;
            
    PyObject* py_V19;
    
        PyArrayObject* V19;
        
            typedef npy_float64 dtype_V19;
            
    PyObject* py_V21;
    
        PyArrayObject* V21;
        
            typedef npy_float64 dtype_V21;
            
    PyObject* py_V23;
    
        PyArrayObject* V23;
        
            typedef npy_float64 dtype_V23;
            
    PyObject* py_V25;
    
        PyArrayObject* V25;
        
            typedef npy_float32 dtype_V25;
            
    PyObject* py_V27;
    
        PyArrayObject* V27;
        
            typedef npy_float32 dtype_V27;
            
    PyObject* py_V29;
    
        PyArrayObject* V29;
        
            typedef npy_bool dtype_V29;
            
    PyObject* py_V31;
    
        PyArrayObject* V31;
        
            typedef npy_float64 dtype_V31;
            
    PyObject* py_V33;
    
        PyArrayObject* V33;
        
            typedef npy_bool dtype_V33;
            
    PyObject* py_V35;
    
        PyArrayObject* V35;
        
            typedef npy_float64 dtype_V35;
            
    PyObject* py_V37;
    
        PyArrayObject* V37;
        
            typedef npy_float64 dtype_V37;
            
    PyObject* py_V39;
    
        PyArrayObject* V39;
        
            typedef npy_float64 dtype_V39;
            
    PyObject* py_V41;
    
        PyArrayObject* V41;
        
            typedef npy_float64 dtype_V41;
            
    PyObject* py_V43;
    
        PyArrayObject* V43;
        
            typedef npy_float64 dtype_V43;
            
    PyObject* py_V45;
    
        PyArrayObject* V45;
        
            typedef npy_float64 dtype_V45;
            
    PyObject* py_V47;
    
        PyArrayObject* V47;
        
            typedef npy_float64 dtype_V47;
            
    PyObject* py_V49;
    
        PyArrayObject* V49;
        
            typedef npy_float64 dtype_V49;
            
    PyObject* py_V51;
    
        PyArrayObject* V51;
        
            typedef npy_float64 dtype_V51;
            
{

    py_V1 = PyList_GET_ITEM(storage_V1, 0);
    {Py_XINCREF(py_V1);}
    
        if (py_V1 == Py_None)
        {
            
        V1 = NULL;
        
        }
        else
        {
            
            V1 = NULL;
            if (py_V1 == Py_None) {
                // We can either fail here or set V1 to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                {
        __failure = 2;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_2;}
            }
            if (!PyArray_Check(py_V1)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                {
        __failure = 2;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_2;}
            }
            // We expect NPY_BOOL
            if (!PyArray_ISALIGNED((PyArrayObject*) py_V1)) {
                PyArrayObject * tmp = (PyArrayObject*) py_V1;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_BOOL), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_BOOL,
                             (long int) PyArray_TYPE((PyArrayObject*) py_V1),
                             (long int) PyArray_NDIM(tmp),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-1] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-1] : -1)
            );
                {
        __failure = 2;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_2;}
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_V1) != NPY_BOOL) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_BOOL) got %d",
                             NPY_BOOL, PyArray_TYPE((PyArrayObject*) py_V1));
                {
        __failure = 2;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_2;}
            }
            
        V1 = (PyArrayObject*)(py_V1);
        Py_XINCREF(V1);
        
        }
        
{

    py_V3 = PyList_GET_ITEM(storage_V3, 0);
    {Py_XINCREF(py_V3);}
    
        if (py_V3 == Py_None)
        {
            
        V3 = NULL;
        
        }
        else
        {
            
            V3 = NULL;
            if (py_V3 == Py_None) {
                // We can either fail here or set V3 to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                {
        __failure = 4;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_4;}
            }
            if (!PyArray_Check(py_V3)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                {
        __failure = 4;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_4;}
            }
            // We expect NPY_BOOL
            if (!PyArray_ISALIGNED((PyArrayObject*) py_V3)) {
                PyArrayObject * tmp = (PyArrayObject*) py_V3;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_BOOL), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_BOOL,
                             (long int) PyArray_TYPE((PyArrayObject*) py_V3),
                             (long int) PyArray_NDIM(tmp),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-1] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-1] : -1)
            );
                {
        __failure = 4;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_4;}
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_V3) != NPY_BOOL) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_BOOL) got %d",
                             NPY_BOOL, PyArray_TYPE((PyArrayObject*) py_V3));
                {
        __failure = 4;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_4;}
            }
            
        V3 = (PyArrayObject*)(py_V3);
        Py_XINCREF(V3);
        
        }
        
{

    py_V5 = PyList_GET_ITEM(storage_V5, 0);
    {Py_XINCREF(py_V5);}
    
        if (py_V5 == Py_None)
        {
            
        V5 = NULL;
        
        }
        else
        {
            
            V5 = NULL;
            if (py_V5 == Py_None) {
                // We can either fail here or set V5 to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                {
        __failure = 6;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_6;}
            }
            if (!PyArray_Check(py_V5)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                {
        __failure = 6;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_6;}
            }
            // We expect NPY_FLOAT64
            if (!PyArray_ISALIGNED((PyArrayObject*) py_V5)) {
                PyArrayObject * tmp = (PyArrayObject*) py_V5;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_FLOAT64), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_FLOAT64,
                             (long int) PyArray_TYPE((PyArrayObject*) py_V5),
                             (long int) PyArray_NDIM(tmp),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-1] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-1] : -1)
            );
                {
        __failure = 6;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_6;}
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_V5) != NPY_FLOAT64) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_FLOAT64) got %d",
                             NPY_FLOAT64, PyArray_TYPE((PyArrayObject*) py_V5));
                {
        __failure = 6;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_6;}
            }
            
        V5 = (PyArrayObject*)(py_V5);
        Py_XINCREF(V5);
        
        }
        
{

    py_V7 = PyList_GET_ITEM(storage_V7, 0);
    {Py_XINCREF(py_V7);}
    
        if (py_V7 == Py_None)
        {
            
        V7 = NULL;
        
        }
        else
        {
            
            V7 = NULL;
            if (py_V7 == Py_None) {
                // We can either fail here or set V7 to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                {
        __failure = 8;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_8;}
            }
            if (!PyArray_Check(py_V7)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                {
        __failure = 8;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_8;}
            }
            // We expect NPY_FLOAT64
            if (!PyArray_ISALIGNED((PyArrayObject*) py_V7)) {
                PyArrayObject * tmp = (PyArrayObject*) py_V7;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_FLOAT64), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_FLOAT64,
                             (long int) PyArray_TYPE((PyArrayObject*) py_V7),
                             (long int) PyArray_NDIM(tmp),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-1] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-1] : -1)
            );
                {
        __failure = 8;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_8;}
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_V7) != NPY_FLOAT64) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_FLOAT64) got %d",
                             NPY_FLOAT64, PyArray_TYPE((PyArrayObject*) py_V7));
                {
        __failure = 8;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_8;}
            }
            
        V7 = (PyArrayObject*)(py_V7);
        Py_XINCREF(V7);
        
        }
        
{

    py_V9 = PyList_GET_ITEM(storage_V9, 0);
    {Py_XINCREF(py_V9);}
    
        if (py_V9 == Py_None)
        {
            
        V9 = NULL;
        
        }
        else
        {
            
            V9 = NULL;
            if (py_V9 == Py_None) {
                // We can either fail here or set V9 to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                {
        __failure = 10;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_10;}
            }
            if (!PyArray_Check(py_V9)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                {
        __failure = 10;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_10;}
            }
            // We expect NPY_FLOAT64
            if (!PyArray_ISALIGNED((PyArrayObject*) py_V9)) {
                PyArrayObject * tmp = (PyArrayObject*) py_V9;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_FLOAT64), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_FLOAT64,
                             (long int) PyArray_TYPE((PyArrayObject*) py_V9),
                             (long int) PyArray_NDIM(tmp),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-1] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-1] : -1)
            );
                {
        __failure = 10;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_10;}
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_V9) != NPY_FLOAT64) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_FLOAT64) got %d",
                             NPY_FLOAT64, PyArray_TYPE((PyArrayObject*) py_V9));
                {
        __failure = 10;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_10;}
            }
            
        V9 = (PyArrayObject*)(py_V9);
        Py_XINCREF(V9);
        
        }
        
{

    py_V11 = PyList_GET_ITEM(storage_V11, 0);
    {Py_XINCREF(py_V11);}
    
        if (py_V11 == Py_None)
        {
            
        V11 = NULL;
        
        }
        else
        {
            
            V11 = NULL;
            if (py_V11 == Py_None) {
                // We can either fail here or set V11 to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                {
        __failure = 12;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_12;}
            }
            if (!PyArray_Check(py_V11)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                {
        __failure = 12;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_12;}
            }
            // We expect NPY_FLOAT64
            if (!PyArray_ISALIGNED((PyArrayObject*) py_V11)) {
                PyArrayObject * tmp = (PyArrayObject*) py_V11;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_FLOAT64), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_FLOAT64,
                             (long int) PyArray_TYPE((PyArrayObject*) py_V11),
                             (long int) PyArray_NDIM(tmp),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-1] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-1] : -1)
            );
                {
        __failure = 12;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_12;}
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_V11) != NPY_FLOAT64) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_FLOAT64) got %d",
                             NPY_FLOAT64, PyArray_TYPE((PyArrayObject*) py_V11));
                {
        __failure = 12;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_12;}
            }
            
        V11 = (PyArrayObject*)(py_V11);
        Py_XINCREF(V11);
        
        }
        
{

    py_V13 = PyList_GET_ITEM(storage_V13, 0);
    {Py_XINCREF(py_V13);}
    
        if (py_V13 == Py_None)
        {
            
        V13 = NULL;
        
        }
        else
        {
            
            V13 = NULL;
            if (py_V13 == Py_None) {
                // We can either fail here or set V13 to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                {
        __failure = 14;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_14;}
            }
            if (!PyArray_Check(py_V13)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                {
        __failure = 14;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_14;}
            }
            // We expect NPY_FLOAT64
            if (!PyArray_ISALIGNED((PyArrayObject*) py_V13)) {
                PyArrayObject * tmp = (PyArrayObject*) py_V13;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_FLOAT64), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_FLOAT64,
                             (long int) PyArray_TYPE((PyArrayObject*) py_V13),
                             (long int) PyArray_NDIM(tmp),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-1] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-1] : -1)
            );
                {
        __failure = 14;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_14;}
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_V13) != NPY_FLOAT64) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_FLOAT64) got %d",
                             NPY_FLOAT64, PyArray_TYPE((PyArrayObject*) py_V13));
                {
        __failure = 14;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_14;}
            }
            
        V13 = (PyArrayObject*)(py_V13);
        Py_XINCREF(V13);
        
        }
        
{

    py_V15 = PyList_GET_ITEM(storage_V15, 0);
    {Py_XINCREF(py_V15);}
    
        if (py_V15 == Py_None)
        {
            
        V15 = NULL;
        
        }
        else
        {
            
            V15 = NULL;
            if (py_V15 == Py_None) {
                // We can either fail here or set V15 to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                {
        __failure = 16;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_16;}
            }
            if (!PyArray_Check(py_V15)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                {
        __failure = 16;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_16;}
            }
            // We expect NPY_FLOAT64
            if (!PyArray_ISALIGNED((PyArrayObject*) py_V15)) {
                PyArrayObject * tmp = (PyArrayObject*) py_V15;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_FLOAT64), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_FLOAT64,
                             (long int) PyArray_TYPE((PyArrayObject*) py_V15),
                             (long int) PyArray_NDIM(tmp),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-1] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-1] : -1)
            );
                {
        __failure = 16;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_16;}
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_V15) != NPY_FLOAT64) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_FLOAT64) got %d",
                             NPY_FLOAT64, PyArray_TYPE((PyArrayObject*) py_V15));
                {
        __failure = 16;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_16;}
            }
            
        V15 = (PyArrayObject*)(py_V15);
        Py_XINCREF(V15);
        
        }
        
{

    py_V17 = PyList_GET_ITEM(storage_V17, 0);
    {Py_XINCREF(py_V17);}
    
        if (py_V17 == Py_None)
        {
            
        V17 = NULL;
        
        }
        else
        {
            
            V17 = NULL;
            if (py_V17 == Py_None) {
                // We can either fail here or set V17 to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                {
        __failure = 18;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_18;}
            }
            if (!PyArray_Check(py_V17)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                {
        __failure = 18;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_18;}
            }
            // We expect NPY_FLOAT64
            if (!PyArray_ISALIGNED((PyArrayObject*) py_V17)) {
                PyArrayObject * tmp = (PyArrayObject*) py_V17;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_FLOAT64), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_FLOAT64,
                             (long int) PyArray_TYPE((PyArrayObject*) py_V17),
                             (long int) PyArray_NDIM(tmp),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-1] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-1] : -1)
            );
                {
        __failure = 18;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_18;}
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_V17) != NPY_FLOAT64) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_FLOAT64) got %d",
                             NPY_FLOAT64, PyArray_TYPE((PyArrayObject*) py_V17));
                {
        __failure = 18;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_18;}
            }
            
        V17 = (PyArrayObject*)(py_V17);
        Py_XINCREF(V17);
        
        }
        
{

    py_V19 = PyList_GET_ITEM(storage_V19, 0);
    {Py_XINCREF(py_V19);}
    
        if (py_V19 == Py_None)
        {
            
        V19 = NULL;
        
        }
        else
        {
            
            V19 = NULL;
            if (py_V19 == Py_None) {
                // We can either fail here or set V19 to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                {
        __failure = 20;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_20;}
            }
            if (!PyArray_Check(py_V19)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                {
        __failure = 20;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_20;}
            }
            // We expect NPY_FLOAT64
            if (!PyArray_ISALIGNED((PyArrayObject*) py_V19)) {
                PyArrayObject * tmp = (PyArrayObject*) py_V19;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_FLOAT64), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_FLOAT64,
                             (long int) PyArray_TYPE((PyArrayObject*) py_V19),
                             (long int) PyArray_NDIM(tmp),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-1] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-1] : -1)
            );
                {
        __failure = 20;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_20;}
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_V19) != NPY_FLOAT64) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_FLOAT64) got %d",
                             NPY_FLOAT64, PyArray_TYPE((PyArrayObject*) py_V19));
                {
        __failure = 20;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_20;}
            }
            
        V19 = (PyArrayObject*)(py_V19);
        Py_XINCREF(V19);
        
        }
        
{

    py_V21 = PyList_GET_ITEM(storage_V21, 0);
    {Py_XINCREF(py_V21);}
    
            V21 = NULL;
            if (py_V21 == Py_None) {
                // We can either fail here or set V21 to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                {
        __failure = 22;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_22;}
            }
            if (!PyArray_Check(py_V21)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                {
        __failure = 22;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_22;}
            }
            // We expect NPY_FLOAT64
            if (!PyArray_ISALIGNED((PyArrayObject*) py_V21)) {
                PyArrayObject * tmp = (PyArrayObject*) py_V21;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_FLOAT64), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_FLOAT64,
                             (long int) PyArray_TYPE((PyArrayObject*) py_V21),
                             (long int) PyArray_NDIM(tmp),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-1] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-1] : -1)
            );
                {
        __failure = 22;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_22;}
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_V21) != NPY_FLOAT64) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_FLOAT64) got %d",
                             NPY_FLOAT64, PyArray_TYPE((PyArrayObject*) py_V21));
                {
        __failure = 22;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_22;}
            }
            
        V21 = (PyArrayObject*)(py_V21);
        Py_XINCREF(V21);
        
{

    py_V23 = PyList_GET_ITEM(storage_V23, 0);
    {Py_XINCREF(py_V23);}
    
            V23 = NULL;
            if (py_V23 == Py_None) {
                // We can either fail here or set V23 to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                {
        __failure = 24;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_24;}
            }
            if (!PyArray_Check(py_V23)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                {
        __failure = 24;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_24;}
            }
            // We expect NPY_FLOAT64
            if (!PyArray_ISALIGNED((PyArrayObject*) py_V23)) {
                PyArrayObject * tmp = (PyArrayObject*) py_V23;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_FLOAT64), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_FLOAT64,
                             (long int) PyArray_TYPE((PyArrayObject*) py_V23),
                             (long int) PyArray_NDIM(tmp),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-1] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-1] : -1)
            );
                {
        __failure = 24;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_24;}
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_V23) != NPY_FLOAT64) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_FLOAT64) got %d",
                             NPY_FLOAT64, PyArray_TYPE((PyArrayObject*) py_V23));
                {
        __failure = 24;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_24;}
            }
            
        V23 = (PyArrayObject*)(py_V23);
        Py_XINCREF(V23);
        
{

    py_V25 = PyList_GET_ITEM(storage_V25, 0);
    {Py_XINCREF(py_V25);}
    
            V25 = NULL;
            if (py_V25 == Py_None) {
                // We can either fail here or set V25 to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                {
        __failure = 26;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_26;}
            }
            if (!PyArray_Check(py_V25)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                {
        __failure = 26;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_26;}
            }
            // We expect NPY_FLOAT32
            if (!PyArray_ISALIGNED((PyArrayObject*) py_V25)) {
                PyArrayObject * tmp = (PyArrayObject*) py_V25;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_FLOAT32), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_FLOAT32,
                             (long int) PyArray_TYPE((PyArrayObject*) py_V25),
                             (long int) PyArray_NDIM(tmp),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-1] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-1] : -1)
            );
                {
        __failure = 26;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_26;}
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_V25) != NPY_FLOAT32) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_FLOAT32) got %d",
                             NPY_FLOAT32, PyArray_TYPE((PyArrayObject*) py_V25));
                {
        __failure = 26;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_26;}
            }
            
        V25 = (PyArrayObject*)(py_V25);
        Py_XINCREF(V25);
        
{

    py_V27 = PyList_GET_ITEM(storage_V27, 0);
    {Py_XINCREF(py_V27);}
    
            V27 = NULL;
            if (py_V27 == Py_None) {
                // We can either fail here or set V27 to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                {
        __failure = 28;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_28;}
            }
            if (!PyArray_Check(py_V27)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                {
        __failure = 28;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_28;}
            }
            // We expect NPY_FLOAT32
            if (!PyArray_ISALIGNED((PyArrayObject*) py_V27)) {
                PyArrayObject * tmp = (PyArrayObject*) py_V27;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_FLOAT32), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_FLOAT32,
                             (long int) PyArray_TYPE((PyArrayObject*) py_V27),
                             (long int) PyArray_NDIM(tmp),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-1] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-1] : -1)
            );
                {
        __failure = 28;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_28;}
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_V27) != NPY_FLOAT32) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_FLOAT32) got %d",
                             NPY_FLOAT32, PyArray_TYPE((PyArrayObject*) py_V27));
                {
        __failure = 28;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_28;}
            }
            
        V27 = (PyArrayObject*)(py_V27);
        Py_XINCREF(V27);
        
{

    py_V29 = PyList_GET_ITEM(storage_V29, 0);
    {Py_XINCREF(py_V29);}
    
            V29 = NULL;
            if (py_V29 == Py_None) {
                // We can either fail here or set V29 to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                {
        __failure = 30;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_30;}
            }
            if (!PyArray_Check(py_V29)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                {
        __failure = 30;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_30;}
            }
            // We expect NPY_BOOL
            if (!PyArray_ISALIGNED((PyArrayObject*) py_V29)) {
                PyArrayObject * tmp = (PyArrayObject*) py_V29;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_BOOL), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_BOOL,
                             (long int) PyArray_TYPE((PyArrayObject*) py_V29),
                             (long int) PyArray_NDIM(tmp),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-1] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-1] : -1)
            );
                {
        __failure = 30;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_30;}
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_V29) != NPY_BOOL) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_BOOL) got %d",
                             NPY_BOOL, PyArray_TYPE((PyArrayObject*) py_V29));
                {
        __failure = 30;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_30;}
            }
            
        V29 = (PyArrayObject*)(py_V29);
        Py_XINCREF(V29);
        
{

    py_V31 = PyList_GET_ITEM(storage_V31, 0);
    {Py_XINCREF(py_V31);}
    
            V31 = NULL;
            if (py_V31 == Py_None) {
                // We can either fail here or set V31 to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                {
        __failure = 32;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_32;}
            }
            if (!PyArray_Check(py_V31)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                {
        __failure = 32;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_32;}
            }
            // We expect NPY_FLOAT64
            if (!PyArray_ISALIGNED((PyArrayObject*) py_V31)) {
                PyArrayObject * tmp = (PyArrayObject*) py_V31;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_FLOAT64), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_FLOAT64,
                             (long int) PyArray_TYPE((PyArrayObject*) py_V31),
                             (long int) PyArray_NDIM(tmp),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-1] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-1] : -1)
            );
                {
        __failure = 32;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_32;}
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_V31) != NPY_FLOAT64) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_FLOAT64) got %d",
                             NPY_FLOAT64, PyArray_TYPE((PyArrayObject*) py_V31));
                {
        __failure = 32;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_32;}
            }
            
        V31 = (PyArrayObject*)(py_V31);
        Py_XINCREF(V31);
        
{

    py_V33 = PyList_GET_ITEM(storage_V33, 0);
    {Py_XINCREF(py_V33);}
    
            V33 = NULL;
            if (py_V33 == Py_None) {
                // We can either fail here or set V33 to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                {
        __failure = 34;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_34;}
            }
            if (!PyArray_Check(py_V33)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                {
        __failure = 34;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_34;}
            }
            // We expect NPY_BOOL
            if (!PyArray_ISALIGNED((PyArrayObject*) py_V33)) {
                PyArrayObject * tmp = (PyArrayObject*) py_V33;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_BOOL), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_BOOL,
                             (long int) PyArray_TYPE((PyArrayObject*) py_V33),
                             (long int) PyArray_NDIM(tmp),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-1] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-1] : -1)
            );
                {
        __failure = 34;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_34;}
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_V33) != NPY_BOOL) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_BOOL) got %d",
                             NPY_BOOL, PyArray_TYPE((PyArrayObject*) py_V33));
                {
        __failure = 34;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_34;}
            }
            
        V33 = (PyArrayObject*)(py_V33);
        Py_XINCREF(V33);
        
{

    py_V35 = PyList_GET_ITEM(storage_V35, 0);
    {Py_XINCREF(py_V35);}
    
            V35 = NULL;
            if (py_V35 == Py_None) {
                // We can either fail here or set V35 to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                {
        __failure = 36;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_36;}
            }
            if (!PyArray_Check(py_V35)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                {
        __failure = 36;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_36;}
            }
            // We expect NPY_FLOAT64
            if (!PyArray_ISALIGNED((PyArrayObject*) py_V35)) {
                PyArrayObject * tmp = (PyArrayObject*) py_V35;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_FLOAT64), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_FLOAT64,
                             (long int) PyArray_TYPE((PyArrayObject*) py_V35),
                             (long int) PyArray_NDIM(tmp),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-1] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-1] : -1)
            );
                {
        __failure = 36;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_36;}
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_V35) != NPY_FLOAT64) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_FLOAT64) got %d",
                             NPY_FLOAT64, PyArray_TYPE((PyArrayObject*) py_V35));
                {
        __failure = 36;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_36;}
            }
            
        V35 = (PyArrayObject*)(py_V35);
        Py_XINCREF(V35);
        
{

    py_V37 = PyList_GET_ITEM(storage_V37, 0);
    {Py_XINCREF(py_V37);}
    
            V37 = NULL;
            if (py_V37 == Py_None) {
                // We can either fail here or set V37 to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                {
        __failure = 38;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_38;}
            }
            if (!PyArray_Check(py_V37)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                {
        __failure = 38;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_38;}
            }
            // We expect NPY_FLOAT64
            if (!PyArray_ISALIGNED((PyArrayObject*) py_V37)) {
                PyArrayObject * tmp = (PyArrayObject*) py_V37;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_FLOAT64), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_FLOAT64,
                             (long int) PyArray_TYPE((PyArrayObject*) py_V37),
                             (long int) PyArray_NDIM(tmp),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-1] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-1] : -1)
            );
                {
        __failure = 38;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_38;}
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_V37) != NPY_FLOAT64) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_FLOAT64) got %d",
                             NPY_FLOAT64, PyArray_TYPE((PyArrayObject*) py_V37));
                {
        __failure = 38;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_38;}
            }
            
        V37 = (PyArrayObject*)(py_V37);
        Py_XINCREF(V37);
        
{

    py_V39 = PyList_GET_ITEM(storage_V39, 0);
    {Py_XINCREF(py_V39);}
    
            V39 = NULL;
            if (py_V39 == Py_None) {
                // We can either fail here or set V39 to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                {
        __failure = 40;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_40;}
            }
            if (!PyArray_Check(py_V39)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                {
        __failure = 40;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_40;}
            }
            // We expect NPY_FLOAT64
            if (!PyArray_ISALIGNED((PyArrayObject*) py_V39)) {
                PyArrayObject * tmp = (PyArrayObject*) py_V39;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_FLOAT64), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_FLOAT64,
                             (long int) PyArray_TYPE((PyArrayObject*) py_V39),
                             (long int) PyArray_NDIM(tmp),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-1] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-1] : -1)
            );
                {
        __failure = 40;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_40;}
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_V39) != NPY_FLOAT64) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_FLOAT64) got %d",
                             NPY_FLOAT64, PyArray_TYPE((PyArrayObject*) py_V39));
                {
        __failure = 40;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_40;}
            }
            
        V39 = (PyArrayObject*)(py_V39);
        Py_XINCREF(V39);
        
{

    py_V41 = PyList_GET_ITEM(storage_V41, 0);
    {Py_XINCREF(py_V41);}
    
            V41 = NULL;
            if (py_V41 == Py_None) {
                // We can either fail here or set V41 to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                {
        __failure = 42;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_42;}
            }
            if (!PyArray_Check(py_V41)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                {
        __failure = 42;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_42;}
            }
            // We expect NPY_FLOAT64
            if (!PyArray_ISALIGNED((PyArrayObject*) py_V41)) {
                PyArrayObject * tmp = (PyArrayObject*) py_V41;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_FLOAT64), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_FLOAT64,
                             (long int) PyArray_TYPE((PyArrayObject*) py_V41),
                             (long int) PyArray_NDIM(tmp),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-1] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-1] : -1)
            );
                {
        __failure = 42;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_42;}
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_V41) != NPY_FLOAT64) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_FLOAT64) got %d",
                             NPY_FLOAT64, PyArray_TYPE((PyArrayObject*) py_V41));
                {
        __failure = 42;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_42;}
            }
            
        V41 = (PyArrayObject*)(py_V41);
        Py_XINCREF(V41);
        
{

    py_V43 = PyList_GET_ITEM(storage_V43, 0);
    {Py_XINCREF(py_V43);}
    
            V43 = NULL;
            if (py_V43 == Py_None) {
                // We can either fail here or set V43 to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                {
        __failure = 44;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_44;}
            }
            if (!PyArray_Check(py_V43)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                {
        __failure = 44;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_44;}
            }
            // We expect NPY_FLOAT64
            if (!PyArray_ISALIGNED((PyArrayObject*) py_V43)) {
                PyArrayObject * tmp = (PyArrayObject*) py_V43;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_FLOAT64), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_FLOAT64,
                             (long int) PyArray_TYPE((PyArrayObject*) py_V43),
                             (long int) PyArray_NDIM(tmp),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-1] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-1] : -1)
            );
                {
        __failure = 44;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_44;}
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_V43) != NPY_FLOAT64) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_FLOAT64) got %d",
                             NPY_FLOAT64, PyArray_TYPE((PyArrayObject*) py_V43));
                {
        __failure = 44;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_44;}
            }
            
        V43 = (PyArrayObject*)(py_V43);
        Py_XINCREF(V43);
        
{

    py_V45 = PyList_GET_ITEM(storage_V45, 0);
    {Py_XINCREF(py_V45);}
    
            V45 = NULL;
            if (py_V45 == Py_None) {
                // We can either fail here or set V45 to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                {
        __failure = 46;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_46;}
            }
            if (!PyArray_Check(py_V45)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                {
        __failure = 46;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_46;}
            }
            // We expect NPY_FLOAT64
            if (!PyArray_ISALIGNED((PyArrayObject*) py_V45)) {
                PyArrayObject * tmp = (PyArrayObject*) py_V45;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_FLOAT64), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_FLOAT64,
                             (long int) PyArray_TYPE((PyArrayObject*) py_V45),
                             (long int) PyArray_NDIM(tmp),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-1] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-1] : -1)
            );
                {
        __failure = 46;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_46;}
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_V45) != NPY_FLOAT64) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_FLOAT64) got %d",
                             NPY_FLOAT64, PyArray_TYPE((PyArrayObject*) py_V45));
                {
        __failure = 46;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_46;}
            }
            
        V45 = (PyArrayObject*)(py_V45);
        Py_XINCREF(V45);
        
{

    py_V47 = PyList_GET_ITEM(storage_V47, 0);
    {Py_XINCREF(py_V47);}
    
            V47 = NULL;
            if (py_V47 == Py_None) {
                // We can either fail here or set V47 to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                {
        __failure = 48;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_48;}
            }
            if (!PyArray_Check(py_V47)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                {
        __failure = 48;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_48;}
            }
            // We expect NPY_FLOAT64
            if (!PyArray_ISALIGNED((PyArrayObject*) py_V47)) {
                PyArrayObject * tmp = (PyArrayObject*) py_V47;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_FLOAT64), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_FLOAT64,
                             (long int) PyArray_TYPE((PyArrayObject*) py_V47),
                             (long int) PyArray_NDIM(tmp),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-1] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-1] : -1)
            );
                {
        __failure = 48;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_48;}
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_V47) != NPY_FLOAT64) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_FLOAT64) got %d",
                             NPY_FLOAT64, PyArray_TYPE((PyArrayObject*) py_V47));
                {
        __failure = 48;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_48;}
            }
            
        V47 = (PyArrayObject*)(py_V47);
        Py_XINCREF(V47);
        
{

    py_V49 = PyList_GET_ITEM(storage_V49, 0);
    {Py_XINCREF(py_V49);}
    
            V49 = NULL;
            if (py_V49 == Py_None) {
                // We can either fail here or set V49 to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                {
        __failure = 50;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_50;}
            }
            if (!PyArray_Check(py_V49)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                {
        __failure = 50;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_50;}
            }
            // We expect NPY_FLOAT64
            if (!PyArray_ISALIGNED((PyArrayObject*) py_V49)) {
                PyArrayObject * tmp = (PyArrayObject*) py_V49;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_FLOAT64), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_FLOAT64,
                             (long int) PyArray_TYPE((PyArrayObject*) py_V49),
                             (long int) PyArray_NDIM(tmp),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-1] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-1] : -1)
            );
                {
        __failure = 50;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_50;}
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_V49) != NPY_FLOAT64) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_FLOAT64) got %d",
                             NPY_FLOAT64, PyArray_TYPE((PyArrayObject*) py_V49));
                {
        __failure = 50;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_50;}
            }
            
        V49 = (PyArrayObject*)(py_V49);
        Py_XINCREF(V49);
        
{

    py_V51 = PyList_GET_ITEM(storage_V51, 0);
    {Py_XINCREF(py_V51);}
    
            V51 = NULL;
            if (py_V51 == Py_None) {
                // We can either fail here or set V51 to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                {
        __failure = 52;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_52;}
            }
            if (!PyArray_Check(py_V51)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                {
        __failure = 52;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_52;}
            }
            // We expect NPY_FLOAT64
            if (!PyArray_ISALIGNED((PyArrayObject*) py_V51)) {
                PyArrayObject * tmp = (PyArrayObject*) py_V51;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %ld "
                             "(NPY_FLOAT64), got non-aligned array of type %ld"
                             " with %ld dimensions, with 3 last dims "
                             "%ld, %ld, %ld"
                             " and 3 last strides %ld %ld, %ld.",
                             (long int) NPY_FLOAT64,
                             (long int) PyArray_TYPE((PyArrayObject*) py_V51),
                             (long int) PyArray_NDIM(tmp),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-1] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-1] : -1)
            );
                {
        __failure = 52;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_52;}
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_V51) != NPY_FLOAT64) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %d (NPY_FLOAT64) got %d",
                             NPY_FLOAT64, PyArray_TYPE((PyArrayObject*) py_V51));
                {
        __failure = 52;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_52;}
            }
            
        V51 = (PyArrayObject*)(py_V51);
        Py_XINCREF(V51);
        
{
// Op class Elemwise
npy_float64* V51_iter;
int V51_jumpx_0;
npy_float64* V49_iter;
npy_intp V49_n0;
ssize_t V49_stride0;
int V49_jump0_0;
npy_float64* V47_iter;
npy_intp V47_n0;
ssize_t V47_stride0;
int V47_jump0_0;
npy_float64* V45_iter;
npy_intp V45_n0;
ssize_t V45_stride0;
int V45_jump0_0;
npy_float64* V43_iter;
npy_intp V43_n0;
ssize_t V43_stride0;
int V43_jump0_0;
npy_float64* V41_iter;
npy_intp V41_n0;
ssize_t V41_stride0;
int V41_jump0_0;
npy_float64* V39_iter;
int V39_jumpx_0;
npy_float64* V37_iter;
npy_intp V37_n0;
ssize_t V37_stride0;
int V37_jump0_0;
npy_float64* V35_iter;
int V35_jumpx_0;
npy_bool* V33_iter;
npy_intp V33_n0;
ssize_t V33_stride0;
int V33_jump0_0;
npy_float64* V31_iter;
int V31_jumpx_0;
npy_bool* V29_iter;
int V29_jumpx_0;
npy_float32* V27_iter;
npy_intp V27_n0;
ssize_t V27_stride0;
int V27_jump0_0;
npy_float32* V25_iter;
int V25_jumpx_0;
npy_float64* V23_iter;
int V23_jumpx_0;
npy_float64* V21_iter;
int V21_jumpx_0;

V51_jumpx_0 = -(0);

if (PyArray_NDIM(V49) < 1) {
    PyErr_SetString(PyExc_ValueError, "Not enough dimensions on input.");
                {
    __failure = 53;
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError,
            "Unexpected error in an Op's C code. "
            "No Python exception was set.");
    }
    goto __label_53;}
}
V49_n0 = PyArray_DIMS(V49)[0];
V49_stride0 = PyArray_STRIDES(V49)[0] / sizeof(npy_float64);
V49_jump0_0 = (V49_stride0) - (0);

if (PyArray_NDIM(V47) < 1) {
    PyErr_SetString(PyExc_ValueError, "Not enough dimensions on input.");
                {
    __failure = 53;
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError,
            "Unexpected error in an Op's C code. "
            "No Python exception was set.");
    }
    goto __label_53;}
}
V47_n0 = PyArray_DIMS(V47)[0];
V47_stride0 = PyArray_STRIDES(V47)[0] / sizeof(npy_float64);
V47_jump0_0 = (V47_stride0) - (0);

if (PyArray_NDIM(V45) < 1) {
    PyErr_SetString(PyExc_ValueError, "Not enough dimensions on input.");
                {
    __failure = 53;
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError,
            "Unexpected error in an Op's C code. "
            "No Python exception was set.");
    }
    goto __label_53;}
}
V45_n0 = PyArray_DIMS(V45)[0];
V45_stride0 = PyArray_STRIDES(V45)[0] / sizeof(npy_float64);
V45_jump0_0 = (V45_stride0) - (0);

if (PyArray_NDIM(V43) < 1) {
    PyErr_SetString(PyExc_ValueError, "Not enough dimensions on input.");
                {
    __failure = 53;
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError,
            "Unexpected error in an Op's C code. "
            "No Python exception was set.");
    }
    goto __label_53;}
}
V43_n0 = PyArray_DIMS(V43)[0];
V43_stride0 = PyArray_STRIDES(V43)[0] / sizeof(npy_float64);
V43_jump0_0 = (V43_stride0) - (0);

if (PyArray_NDIM(V41) < 1) {
    PyErr_SetString(PyExc_ValueError, "Not enough dimensions on input.");
                {
    __failure = 53;
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError,
            "Unexpected error in an Op's C code. "
            "No Python exception was set.");
    }
    goto __label_53;}
}
V41_n0 = PyArray_DIMS(V41)[0];
V41_stride0 = PyArray_STRIDES(V41)[0] / sizeof(npy_float64);
V41_jump0_0 = (V41_stride0) - (0);
V39_jumpx_0 = -(0);

if (PyArray_NDIM(V37) < 1) {
    PyErr_SetString(PyExc_ValueError, "Not enough dimensions on input.");
                {
    __failure = 53;
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError,
            "Unexpected error in an Op's C code. "
            "No Python exception was set.");
    }
    goto __label_53;}
}
V37_n0 = PyArray_DIMS(V37)[0];
V37_stride0 = PyArray_STRIDES(V37)[0] / sizeof(npy_float64);
V37_jump0_0 = (V37_stride0) - (0);
V35_jumpx_0 = -(0);

if (PyArray_NDIM(V33) < 1) {
    PyErr_SetString(PyExc_ValueError, "Not enough dimensions on input.");
                {
    __failure = 53;
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError,
            "Unexpected error in an Op's C code. "
            "No Python exception was set.");
    }
    goto __label_53;}
}
V33_n0 = PyArray_DIMS(V33)[0];
V33_stride0 = PyArray_STRIDES(V33)[0] / sizeof(npy_bool);
V33_jump0_0 = (V33_stride0) - (0);
V31_jumpx_0 = -(0);
V29_jumpx_0 = -(0);

if (PyArray_NDIM(V27) < 1) {
    PyErr_SetString(PyExc_ValueError, "Not enough dimensions on input.");
                {
    __failure = 53;
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError,
            "Unexpected error in an Op's C code. "
            "No Python exception was set.");
    }
    goto __label_53;}
}
V27_n0 = PyArray_DIMS(V27)[0];
V27_stride0 = PyArray_STRIDES(V27)[0] / sizeof(npy_float32);
V27_jump0_0 = (V27_stride0) - (0);
V25_jumpx_0 = -(0);
V23_jumpx_0 = -(0);
V21_jumpx_0 = -(0);

        if (V49_n0 != V47_n0)
        {
            if (V49_n0 == 1 || V47_n0 == 1)
            {
                PyErr_Format(PyExc_ValueError, "Runtime broadcasting not allowed. One input had a distinct dimension length of 1, but was not marked as broadcastable: (input[%%i].shape[%%i] = %%lld, input[%%i].shape[%%i] = %%lld). If broadcasting was intended, use `specify_broadcastable` on the relevant input.",
               1,
               0,
               (long long int) V49_n0,
               2,
               0,
               (long long int) V47_n0
                );
            } else {
                PyErr_Format(PyExc_ValueError, "Input dimension mismatch: (input[%%i].shape[%%i] = %%lld, input[%%i].shape[%%i] = %%lld)",
                   1,
                   0,
                   (long long int) V49_n0,
                   2,
                   0,
                   (long long int) V47_n0
                );
            }
            {
__failure = 53;
if (!PyErr_Occurred()) {
    PyErr_SetString(PyExc_RuntimeError,
        "Unexpected error in an Op's C code. "
        "No Python exception was set.");
}
goto __label_53;}
        }

        if (V49_n0 != V45_n0)
        {
            if (V49_n0 == 1 || V45_n0 == 1)
            {
                PyErr_Format(PyExc_ValueError, "Runtime broadcasting not allowed. One input had a distinct dimension length of 1, but was not marked as broadcastable: (input[%%i].shape[%%i] = %%lld, input[%%i].shape[%%i] = %%lld). If broadcasting was intended, use `specify_broadcastable` on the relevant input.",
               1,
               0,
               (long long int) V49_n0,
               3,
               0,
               (long long int) V45_n0
                );
            } else {
                PyErr_Format(PyExc_ValueError, "Input dimension mismatch: (input[%%i].shape[%%i] = %%lld, input[%%i].shape[%%i] = %%lld)",
                   1,
                   0,
                   (long long int) V49_n0,
                   3,
                   0,
                   (long long int) V45_n0
                );
            }
            {
__failure = 53;
if (!PyErr_Occurred()) {
    PyErr_SetString(PyExc_RuntimeError,
        "Unexpected error in an Op's C code. "
        "No Python exception was set.");
}
goto __label_53;}
        }

        if (V49_n0 != V43_n0)
        {
            if (V49_n0 == 1 || V43_n0 == 1)
            {
                PyErr_Format(PyExc_ValueError, "Runtime broadcasting not allowed. One input had a distinct dimension length of 1, but was not marked as broadcastable: (input[%%i].shape[%%i] = %%lld, input[%%i].shape[%%i] = %%lld). If broadcasting was intended, use `specify_broadcastable` on the relevant input.",
               1,
               0,
               (long long int) V49_n0,
               4,
               0,
               (long long int) V43_n0
                );
            } else {
                PyErr_Format(PyExc_ValueError, "Input dimension mismatch: (input[%%i].shape[%%i] = %%lld, input[%%i].shape[%%i] = %%lld)",
                   1,
                   0,
                   (long long int) V49_n0,
                   4,
                   0,
                   (long long int) V43_n0
                );
            }
            {
__failure = 53;
if (!PyErr_Occurred()) {
    PyErr_SetString(PyExc_RuntimeError,
        "Unexpected error in an Op's C code. "
        "No Python exception was set.");
}
goto __label_53;}
        }

        if (V49_n0 != V41_n0)
        {
            if (V49_n0 == 1 || V41_n0 == 1)
            {
                PyErr_Format(PyExc_ValueError, "Runtime broadcasting not allowed. One input had a distinct dimension length of 1, but was not marked as broadcastable: (input[%%i].shape[%%i] = %%lld, input[%%i].shape[%%i] = %%lld). If broadcasting was intended, use `specify_broadcastable` on the relevant input.",
               1,
               0,
               (long long int) V49_n0,
               5,
               0,
               (long long int) V41_n0
                );
            } else {
                PyErr_Format(PyExc_ValueError, "Input dimension mismatch: (input[%%i].shape[%%i] = %%lld, input[%%i].shape[%%i] = %%lld)",
                   1,
                   0,
                   (long long int) V49_n0,
                   5,
                   0,
                   (long long int) V41_n0
                );
            }
            {
__failure = 53;
if (!PyErr_Occurred()) {
    PyErr_SetString(PyExc_RuntimeError,
        "Unexpected error in an Op's C code. "
        "No Python exception was set.");
}
goto __label_53;}
        }

        if (V49_n0 != V37_n0)
        {
            if (V49_n0 == 1 || V37_n0 == 1)
            {
                PyErr_Format(PyExc_ValueError, "Runtime broadcasting not allowed. One input had a distinct dimension length of 1, but was not marked as broadcastable: (input[%%i].shape[%%i] = %%lld, input[%%i].shape[%%i] = %%lld). If broadcasting was intended, use `specify_broadcastable` on the relevant input.",
               1,
               0,
               (long long int) V49_n0,
               7,
               0,
               (long long int) V37_n0
                );
            } else {
                PyErr_Format(PyExc_ValueError, "Input dimension mismatch: (input[%%i].shape[%%i] = %%lld, input[%%i].shape[%%i] = %%lld)",
                   1,
                   0,
                   (long long int) V49_n0,
                   7,
                   0,
                   (long long int) V37_n0
                );
            }
            {
__failure = 53;
if (!PyErr_Occurred()) {
    PyErr_SetString(PyExc_RuntimeError,
        "Unexpected error in an Op's C code. "
        "No Python exception was set.");
}
goto __label_53;}
        }

        if (V49_n0 != V33_n0)
        {
            if (V49_n0 == 1 || V33_n0 == 1)
            {
                PyErr_Format(PyExc_ValueError, "Runtime broadcasting not allowed. One input had a distinct dimension length of 1, but was not marked as broadcastable: (input[%%i].shape[%%i] = %%lld, input[%%i].shape[%%i] = %%lld). If broadcasting was intended, use `specify_broadcastable` on the relevant input.",
               1,
               0,
               (long long int) V49_n0,
               9,
               0,
               (long long int) V33_n0
                );
            } else {
                PyErr_Format(PyExc_ValueError, "Input dimension mismatch: (input[%%i].shape[%%i] = %%lld, input[%%i].shape[%%i] = %%lld)",
                   1,
                   0,
                   (long long int) V49_n0,
                   9,
                   0,
                   (long long int) V33_n0
                );
            }
            {
__failure = 53;
if (!PyErr_Occurred()) {
    PyErr_SetString(PyExc_RuntimeError,
        "Unexpected error in an Op's C code. "
        "No Python exception was set.");
}
goto __label_53;}
        }

        if (V49_n0 != V27_n0)
        {
            if (V49_n0 == 1 || V27_n0 == 1)
            {
                PyErr_Format(PyExc_ValueError, "Runtime broadcasting not allowed. One input had a distinct dimension length of 1, but was not marked as broadcastable: (input[%%i].shape[%%i] = %%lld, input[%%i].shape[%%i] = %%lld). If broadcasting was intended, use `specify_broadcastable` on the relevant input.",
               1,
               0,
               (long long int) V49_n0,
               12,
               0,
               (long long int) V27_n0
                );
            } else {
                PyErr_Format(PyExc_ValueError, "Input dimension mismatch: (input[%%i].shape[%%i] = %%lld, input[%%i].shape[%%i] = %%lld)",
                   1,
                   0,
                   (long long int) V49_n0,
                   12,
                   0,
                   (long long int) V27_n0
                );
            }
            {
__failure = 53;
if (!PyErr_Occurred()) {
    PyErr_SetString(PyExc_RuntimeError,
        "Unexpected error in an Op's C code. "
        "No Python exception was set.");
}
goto __label_53;}
        }

npy_bool* V1_iter;
npy_intp V1_n0;
ssize_t V1_stride0;
int V1_jump0_0;

{
    npy_intp dims[1];
    dims[0] = V49_n0;

    if (!V1) {
        V1 = (PyArrayObject*)PyArray_EMPTY(1,
                                              dims,
                                              NPY_BOOL,
                                              PyArray_ISFORTRAN(V49) && PyArray_ISFORTRAN(V47) && PyArray_ISFORTRAN(V45) && PyArray_ISFORTRAN(V43) && PyArray_ISFORTRAN(V41) && PyArray_ISFORTRAN(V37) && PyArray_ISFORTRAN(V33) && PyArray_ISFORTRAN(V27));
    }
    else {
        PyArray_Dims new_dims;
        new_dims.len = 1;
        new_dims.ptr = dims;
        PyObject* success = PyArray_Resize(V1, &new_dims, 0, NPY_CORDER);
        if (!success) {
            // If we can't resize the ndarray we have we can allocate a new one.
            PyErr_Clear();
            Py_XDECREF(V1);
            V1 = (PyArrayObject*)PyArray_EMPTY(1, dims, NPY_BOOL, 0);
        } else {
            Py_DECREF(success);
        }
    }
    if (!V1) {
        {
__failure = 53;
if (!PyErr_Occurred()) {
    PyErr_SetString(PyExc_RuntimeError,
        "Unexpected error in an Op's C code. "
        "No Python exception was set.");
}
goto __label_53;}
    }
}

if (PyArray_NDIM(V1) < 1) {
    PyErr_SetString(PyExc_ValueError, "Not enough dimensions on input.");
                {
    __failure = 53;
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError,
            "Unexpected error in an Op's C code. "
            "No Python exception was set.");
    }
    goto __label_53;}
}
V1_n0 = PyArray_DIMS(V1)[0];
V1_stride0 = PyArray_STRIDES(V1)[0] / sizeof(npy_bool);
V1_jump0_0 = (V1_stride0) - (0);
npy_bool* V3_iter;
npy_intp V3_n0;
ssize_t V3_stride0;
int V3_jump0_0;

{
    npy_intp dims[1];
    dims[0] = V49_n0;

    if (!V3) {
        V3 = (PyArrayObject*)PyArray_EMPTY(1,
                                              dims,
                                              NPY_BOOL,
                                              PyArray_ISFORTRAN(V49) && PyArray_ISFORTRAN(V47) && PyArray_ISFORTRAN(V45) && PyArray_ISFORTRAN(V43) && PyArray_ISFORTRAN(V41) && PyArray_ISFORTRAN(V37) && PyArray_ISFORTRAN(V33) && PyArray_ISFORTRAN(V27));
    }
    else {
        PyArray_Dims new_dims;
        new_dims.len = 1;
        new_dims.ptr = dims;
        PyObject* success = PyArray_Resize(V3, &new_dims, 0, NPY_CORDER);
        if (!success) {
            // If we can't resize the ndarray we have we can allocate a new one.
            PyErr_Clear();
            Py_XDECREF(V3);
            V3 = (PyArrayObject*)PyArray_EMPTY(1, dims, NPY_BOOL, 0);
        } else {
            Py_DECREF(success);
        }
    }
    if (!V3) {
        {
__failure = 53;
if (!PyErr_Occurred()) {
    PyErr_SetString(PyExc_RuntimeError,
        "Unexpected error in an Op's C code. "
        "No Python exception was set.");
}
goto __label_53;}
    }
}

if (PyArray_NDIM(V3) < 1) {
    PyErr_SetString(PyExc_ValueError, "Not enough dimensions on input.");
                {
    __failure = 53;
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError,
            "Unexpected error in an Op's C code. "
            "No Python exception was set.");
    }
    goto __label_53;}
}
V3_n0 = PyArray_DIMS(V3)[0];
V3_stride0 = PyArray_STRIDES(V3)[0] / sizeof(npy_bool);
V3_jump0_0 = (V3_stride0) - (0);
npy_float64* V9_iter;
npy_intp V9_n0;
ssize_t V9_stride0;
int V9_jump0_0;

{
    npy_intp dims[1];
    dims[0] = V49_n0;

    if (!V9) {
        V9 = (PyArrayObject*)PyArray_EMPTY(1,
                                              dims,
                                              NPY_FLOAT64,
                                              PyArray_ISFORTRAN(V49) && PyArray_ISFORTRAN(V47) && PyArray_ISFORTRAN(V45) && PyArray_ISFORTRAN(V43) && PyArray_ISFORTRAN(V41) && PyArray_ISFORTRAN(V37) && PyArray_ISFORTRAN(V33) && PyArray_ISFORTRAN(V27));
    }
    else {
        PyArray_Dims new_dims;
        new_dims.len = 1;
        new_dims.ptr = dims;
        PyObject* success = PyArray_Resize(V9, &new_dims, 0, NPY_CORDER);
        if (!success) {
            // If we can't resize the ndarray we have we can allocate a new one.
            PyErr_Clear();
            Py_XDECREF(V9);
            V9 = (PyArrayObject*)PyArray_EMPTY(1, dims, NPY_FLOAT64, 0);
        } else {
            Py_DECREF(success);
        }
    }
    if (!V9) {
        {
__failure = 53;
if (!PyErr_Occurred()) {
    PyErr_SetString(PyExc_RuntimeError,
        "Unexpected error in an Op's C code. "
        "No Python exception was set.");
}
goto __label_53;}
    }
}

if (PyArray_NDIM(V9) < 1) {
    PyErr_SetString(PyExc_ValueError, "Not enough dimensions on input.");
                {
    __failure = 53;
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError,
            "Unexpected error in an Op's C code. "
            "No Python exception was set.");
    }
    goto __label_53;}
}
V9_n0 = PyArray_DIMS(V9)[0];
V9_stride0 = PyArray_STRIDES(V9)[0] / sizeof(npy_float64);
V9_jump0_0 = (V9_stride0) - (0);
npy_float64* V11_iter;
npy_intp V11_n0;
ssize_t V11_stride0;
int V11_jump0_0;

{
    npy_intp dims[1];
    dims[0] = V49_n0;

    if (!V11) {
        V11 = (PyArrayObject*)PyArray_EMPTY(1,
                                              dims,
                                              NPY_FLOAT64,
                                              PyArray_ISFORTRAN(V49) && PyArray_ISFORTRAN(V47) && PyArray_ISFORTRAN(V45) && PyArray_ISFORTRAN(V43) && PyArray_ISFORTRAN(V41) && PyArray_ISFORTRAN(V37) && PyArray_ISFORTRAN(V33) && PyArray_ISFORTRAN(V27));
    }
    else {
        PyArray_Dims new_dims;
        new_dims.len = 1;
        new_dims.ptr = dims;
        PyObject* success = PyArray_Resize(V11, &new_dims, 0, NPY_CORDER);
        if (!success) {
            // If we can't resize the ndarray we have we can allocate a new one.
            PyErr_Clear();
            Py_XDECREF(V11);
            V11 = (PyArrayObject*)PyArray_EMPTY(1, dims, NPY_FLOAT64, 0);
        } else {
            Py_DECREF(success);
        }
    }
    if (!V11) {
        {
__failure = 53;
if (!PyErr_Occurred()) {
    PyErr_SetString(PyExc_RuntimeError,
        "Unexpected error in an Op's C code. "
        "No Python exception was set.");
}
goto __label_53;}
    }
}

if (PyArray_NDIM(V11) < 1) {
    PyErr_SetString(PyExc_ValueError, "Not enough dimensions on input.");
                {
    __failure = 53;
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError,
            "Unexpected error in an Op's C code. "
            "No Python exception was set.");
    }
    goto __label_53;}
}
V11_n0 = PyArray_DIMS(V11)[0];
V11_stride0 = PyArray_STRIDES(V11)[0] / sizeof(npy_float64);
V11_jump0_0 = (V11_stride0) - (0);
npy_float64* V13_iter;
npy_intp V13_n0;
ssize_t V13_stride0;
int V13_jump0_0;

{
    npy_intp dims[1];
    dims[0] = V49_n0;

    if (!V13) {
        V13 = (PyArrayObject*)PyArray_EMPTY(1,
                                              dims,
                                              NPY_FLOAT64,
                                              PyArray_ISFORTRAN(V49) && PyArray_ISFORTRAN(V47) && PyArray_ISFORTRAN(V45) && PyArray_ISFORTRAN(V43) && PyArray_ISFORTRAN(V41) && PyArray_ISFORTRAN(V37) && PyArray_ISFORTRAN(V33) && PyArray_ISFORTRAN(V27));
    }
    else {
        PyArray_Dims new_dims;
        new_dims.len = 1;
        new_dims.ptr = dims;
        PyObject* success = PyArray_Resize(V13, &new_dims, 0, NPY_CORDER);
        if (!success) {
            // If we can't resize the ndarray we have we can allocate a new one.
            PyErr_Clear();
            Py_XDECREF(V13);
            V13 = (PyArrayObject*)PyArray_EMPTY(1, dims, NPY_FLOAT64, 0);
        } else {
            Py_DECREF(success);
        }
    }
    if (!V13) {
        {
__failure = 53;
if (!PyErr_Occurred()) {
    PyErr_SetString(PyExc_RuntimeError,
        "Unexpected error in an Op's C code. "
        "No Python exception was set.");
}
goto __label_53;}
    }
}

if (PyArray_NDIM(V13) < 1) {
    PyErr_SetString(PyExc_ValueError, "Not enough dimensions on input.");
                {
    __failure = 53;
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError,
            "Unexpected error in an Op's C code. "
            "No Python exception was set.");
    }
    goto __label_53;}
}
V13_n0 = PyArray_DIMS(V13)[0];
V13_stride0 = PyArray_STRIDES(V13)[0] / sizeof(npy_float64);
V13_jump0_0 = (V13_stride0) - (0);
npy_float64* V15_iter;
npy_intp V15_n0;
ssize_t V15_stride0;
int V15_jump0_0;

{
    npy_intp dims[1];
    dims[0] = V49_n0;

    if (!V15) {
        V15 = (PyArrayObject*)PyArray_EMPTY(1,
                                              dims,
                                              NPY_FLOAT64,
                                              PyArray_ISFORTRAN(V49) && PyArray_ISFORTRAN(V47) && PyArray_ISFORTRAN(V45) && PyArray_ISFORTRAN(V43) && PyArray_ISFORTRAN(V41) && PyArray_ISFORTRAN(V37) && PyArray_ISFORTRAN(V33) && PyArray_ISFORTRAN(V27));
    }
    else {
        PyArray_Dims new_dims;
        new_dims.len = 1;
        new_dims.ptr = dims;
        PyObject* success = PyArray_Resize(V15, &new_dims, 0, NPY_CORDER);
        if (!success) {
            // If we can't resize the ndarray we have we can allocate a new one.
            PyErr_Clear();
            Py_XDECREF(V15);
            V15 = (PyArrayObject*)PyArray_EMPTY(1, dims, NPY_FLOAT64, 0);
        } else {
            Py_DECREF(success);
        }
    }
    if (!V15) {
        {
__failure = 53;
if (!PyErr_Occurred()) {
    PyErr_SetString(PyExc_RuntimeError,
        "Unexpected error in an Op's C code. "
        "No Python exception was set.");
}
goto __label_53;}
    }
}

if (PyArray_NDIM(V15) < 1) {
    PyErr_SetString(PyExc_ValueError, "Not enough dimensions on input.");
                {
    __failure = 53;
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError,
            "Unexpected error in an Op's C code. "
            "No Python exception was set.");
    }
    goto __label_53;}
}
V15_n0 = PyArray_DIMS(V15)[0];
V15_stride0 = PyArray_STRIDES(V15)[0] / sizeof(npy_float64);
V15_jump0_0 = (V15_stride0) - (0);
npy_float64* V17_iter;
npy_intp V17_n0;
ssize_t V17_stride0;
int V17_jump0_0;

{
    npy_intp dims[1];
    dims[0] = V49_n0;

    if (!V17) {
        V17 = (PyArrayObject*)PyArray_EMPTY(1,
                                              dims,
                                              NPY_FLOAT64,
                                              PyArray_ISFORTRAN(V49) && PyArray_ISFORTRAN(V47) && PyArray_ISFORTRAN(V45) && PyArray_ISFORTRAN(V43) && PyArray_ISFORTRAN(V41) && PyArray_ISFORTRAN(V37) && PyArray_ISFORTRAN(V33) && PyArray_ISFORTRAN(V27));
    }
    else {
        PyArray_Dims new_dims;
        new_dims.len = 1;
        new_dims.ptr = dims;
        PyObject* success = PyArray_Resize(V17, &new_dims, 0, NPY_CORDER);
        if (!success) {
            // If we can't resize the ndarray we have we can allocate a new one.
            PyErr_Clear();
            Py_XDECREF(V17);
            V17 = (PyArrayObject*)PyArray_EMPTY(1, dims, NPY_FLOAT64, 0);
        } else {
            Py_DECREF(success);
        }
    }
    if (!V17) {
        {
__failure = 53;
if (!PyErr_Occurred()) {
    PyErr_SetString(PyExc_RuntimeError,
        "Unexpected error in an Op's C code. "
        "No Python exception was set.");
}
goto __label_53;}
    }
}

if (PyArray_NDIM(V17) < 1) {
    PyErr_SetString(PyExc_ValueError, "Not enough dimensions on input.");
                {
    __failure = 53;
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError,
            "Unexpected error in an Op's C code. "
            "No Python exception was set.");
    }
    goto __label_53;}
}
V17_n0 = PyArray_DIMS(V17)[0];
V17_stride0 = PyArray_STRIDES(V17)[0] / sizeof(npy_float64);
V17_jump0_0 = (V17_stride0) - (0);
npy_float64* V19_iter;
npy_intp V19_n0;
ssize_t V19_stride0;
int V19_jump0_0;

{
    npy_intp dims[1];
    dims[0] = V49_n0;

    if (!V19) {
        V19 = (PyArrayObject*)PyArray_EMPTY(1,
                                              dims,
                                              NPY_FLOAT64,
                                              PyArray_ISFORTRAN(V49) && PyArray_ISFORTRAN(V47) && PyArray_ISFORTRAN(V45) && PyArray_ISFORTRAN(V43) && PyArray_ISFORTRAN(V41) && PyArray_ISFORTRAN(V37) && PyArray_ISFORTRAN(V33) && PyArray_ISFORTRAN(V27));
    }
    else {
        PyArray_Dims new_dims;
        new_dims.len = 1;
        new_dims.ptr = dims;
        PyObject* success = PyArray_Resize(V19, &new_dims, 0, NPY_CORDER);
        if (!success) {
            // If we can't resize the ndarray we have we can allocate a new one.
            PyErr_Clear();
            Py_XDECREF(V19);
            V19 = (PyArrayObject*)PyArray_EMPTY(1, dims, NPY_FLOAT64, 0);
        } else {
            Py_DECREF(success);
        }
    }
    if (!V19) {
        {
__failure = 53;
if (!PyErr_Occurred()) {
    PyErr_SetString(PyExc_RuntimeError,
        "Unexpected error in an Op's C code. "
        "No Python exception was set.");
}
goto __label_53;}
    }
}

if (PyArray_NDIM(V19) < 1) {
    PyErr_SetString(PyExc_ValueError, "Not enough dimensions on input.");
                {
    __failure = 53;
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError,
            "Unexpected error in an Op's C code. "
            "No Python exception was set.");
    }
    goto __label_53;}
}
V19_n0 = PyArray_DIMS(V19)[0];
V19_stride0 = PyArray_STRIDES(V19)[0] / sizeof(npy_float64);
V19_jump0_0 = (V19_stride0) - (0);

            if (V5) {
                Py_XDECREF(V5);
            }
            V5 = V47;
            Py_XINCREF(V5);
            
            if (V7) {
                Py_XDECREF(V7);
            }
            V7 = V45;
            Py_XINCREF(V7);
            
{
        V51_iter = (npy_float64*)(PyArray_DATA(V51));
V49_iter = (npy_float64*)(PyArray_DATA(V49));
V47_iter = (npy_float64*)(PyArray_DATA(V47));
V45_iter = (npy_float64*)(PyArray_DATA(V45));
V43_iter = (npy_float64*)(PyArray_DATA(V43));
V41_iter = (npy_float64*)(PyArray_DATA(V41));
V39_iter = (npy_float64*)(PyArray_DATA(V39));
V37_iter = (npy_float64*)(PyArray_DATA(V37));
V35_iter = (npy_float64*)(PyArray_DATA(V35));
V33_iter = (npy_bool*)(PyArray_DATA(V33));
V31_iter = (npy_float64*)(PyArray_DATA(V31));
V29_iter = (npy_bool*)(PyArray_DATA(V29));
V27_iter = (npy_float32*)(PyArray_DATA(V27));
V25_iter = (npy_float32*)(PyArray_DATA(V25));
V23_iter = (npy_float64*)(PyArray_DATA(V23));
V21_iter = (npy_float64*)(PyArray_DATA(V21));
V1_iter = (npy_bool*)(PyArray_DATA(V1));
V3_iter = (npy_bool*)(PyArray_DATA(V3));
V9_iter = (npy_float64*)(PyArray_DATA(V9));
V11_iter = (npy_float64*)(PyArray_DATA(V11));
V13_iter = (npy_float64*)(PyArray_DATA(V13));
V15_iter = (npy_float64*)(PyArray_DATA(V15));
V17_iter = (npy_float64*)(PyArray_DATA(V17));
V19_iter = (npy_float64*)(PyArray_DATA(V19));

        for (int ITER_0 = 0; ITER_0<V19_n0; ITER_0++) {
            npy_float64 &V51_i = * ( V51_iter + ITER_0 * V51_jumpx_0 );
npy_float64 &V49_i = * ( V49_iter + ITER_0 * V49_jump0_0 );
npy_float64 &V47_i = * ( V47_iter + ITER_0 * V47_jump0_0 );
npy_float64 &V45_i = * ( V45_iter + ITER_0 * V45_jump0_0 );
npy_float64 &V43_i = * ( V43_iter + ITER_0 * V43_jump0_0 );
npy_float64 &V41_i = * ( V41_iter + ITER_0 * V41_jump0_0 );
npy_float64 &V39_i = * ( V39_iter + ITER_0 * V39_jumpx_0 );
npy_float64 &V37_i = * ( V37_iter + ITER_0 * V37_jump0_0 );
npy_float64 &V35_i = * ( V35_iter + ITER_0 * V35_jumpx_0 );
npy_bool &V33_i = * ( V33_iter + ITER_0 * V33_jump0_0 );
npy_float64 &V31_i = * ( V31_iter + ITER_0 * V31_jumpx_0 );
npy_bool &V29_i = * ( V29_iter + ITER_0 * V29_jumpx_0 );
npy_float32 &V27_i = * ( V27_iter + ITER_0 * V27_jump0_0 );
npy_float32 &V25_i = * ( V25_iter + ITER_0 * V25_jumpx_0 );
npy_float64 &V23_i = * ( V23_iter + ITER_0 * V23_jumpx_0 );
npy_float64 &V21_i = * ( V21_iter + ITER_0 * V21_jumpx_0 );
npy_bool &V1_i = * ( V1_iter + ITER_0 * V1_jump0_0 );
npy_bool &V3_i = * ( V3_iter + ITER_0 * V3_jump0_0 );
npy_float64 &V9_i = * ( V9_iter + ITER_0 * V9_jump0_0 );
npy_float64 &V11_i = * ( V11_iter + ITER_0 * V11_jump0_0 );
npy_float64 &V13_i = * ( V13_iter + ITER_0 * V13_jump0_0 );
npy_float64 &V15_i = * ( V15_iter + ITER_0 * V15_jump0_0 );
npy_float64 &V17_i = * ( V17_iter + ITER_0 * V17_jump0_0 );
npy_float64 &V19_i = * ( V19_iter + ITER_0 * V19_jump0_0 );

            
        {
            #define V5_i V47_i
#define V7_i V45_i

            {
npy_float64 V53_tmp1;
V53_tmp1 = V51_i * V49_i;
npy_float64 V53_tmp2;
V53_tmp2 = V47_i + V53_tmp1 + V45_i + V43_i;
npy_float64 V53_tmp3;
V53_tmp3 = exp((npy_float64)V53_tmp2);
npy_bool V53_tmp4;
V53_tmp4 = (V53_tmp3 > (0));
npy_bool V53_tmp5;
V53_tmp5 = (V53_tmp3 >= (0));
npy_float64 V53_tmp6;
V53_tmp6 = V53_tmp3 + V39_i;
npy_float64 V53_tmp7;
V53_tmp7 = V39_i / V53_tmp6;
npy_float64 V53_tmp8;
V53_tmp8 = log((npy_float64)V53_tmp7);
npy_float64 V53_tmp9;
V53_tmp9 = V29_i ? (0.0) : V53_tmp8;
npy_bool V53_tmp10;
V53_tmp10 = (V53_tmp7 == (0));
npy_float64 V53_tmp11;
V53_tmp11 = V53_tmp10 ? (0.0) : V53_tmp9;
npy_float64 V53_tmp12;
V53_tmp12 = V53_tmp10 ? (0.0) : V23_i;
npy_float64 V53_tmp13;
V53_tmp13 = V53_tmp10 ? (0.0) : V31_i;
npy_float64 V53_tmp14;
V53_tmp14 = -V53_tmp13;
npy_float64 V53_tmp15;
V53_tmp15 = V53_tmp14 / V53_tmp6;
npy_float64 V53_tmp16;
V53_tmp16 = V29_i ? (0.0) : V41_i;
npy_float64 V53_tmp17;
V53_tmp17 = V53_tmp3 / V53_tmp6;
npy_bool V53_tmp18;
V53_tmp18 = (V53_tmp17 == (0));
npy_float64 V53_tmp19;
V53_tmp19 = V53_tmp18 ? (0.0) : V53_tmp16;
npy_float64 V53_tmp20;
V53_tmp20 = -V53_tmp19;
npy_float64 V53_tmp21;
V53_tmp21 = V53_tmp20 / V53_tmp6;
npy_float64 V53_tmp22;
V53_tmp22 = V53_tmp21 + V53_tmp15;
npy_float64 V53_tmp23;
V53_tmp23 = V29_i ? V41_i : (0.0);
npy_bool V53_tmp24;
V53_tmp24 = (V53_tmp3 == (0));
npy_bool V53_tmp25;
V53_tmp25 = V53_tmp24 && V33_i;
npy_float64 V53_tmp26;
V53_tmp26 = V53_tmp25 ? (0.0) : V53_tmp23;
npy_float64 V53_tmp27;
V53_tmp27 = V53_tmp24 ? (0.0) : V53_tmp26;
npy_float64 V53_tmp28;
V53_tmp28 = V53_tmp25 ? (0.0) : V35_i;
npy_float64 V53_tmp29;
V53_tmp29 = V41_i / V53_tmp3;
npy_float64 V53_tmp30;
V53_tmp30 = V29_i ? (0.0) : V53_tmp29;
npy_float64 V53_tmp31;
V53_tmp31 = V53_tmp18 ? (0.0) : V53_tmp30;
npy_float64 V53_tmp32;
V53_tmp32 = V53_tmp31 + V53_tmp21 + V53_tmp15;
npy_float64 V53_tmp33;
V53_tmp33 = V53_tmp32 - V53_tmp28;
npy_float64 V53_tmp34;
V53_tmp34 = V53_tmp33 * V53_tmp3;
npy_float64 V53_tmp35;
V53_tmp35 = V53_tmp34 + V53_tmp27;
npy_float64 V53_tmp36;
V53_tmp36 = V37_i + V53_tmp3;
npy_float64 V53_tmp37;
V53_tmp37 = V41_i * V53_tmp2;
npy_float64 V53_tmp38;
V53_tmp38 = V53_tmp24 ? V27_i : V53_tmp37;
npy_float64 V53_tmp39;
V53_tmp39 = V53_tmp38 - V53_tmp36;
npy_float64 V53_tmp40;
V53_tmp40 = V53_tmp25 ? (0) : V53_tmp39;
npy_float64 V53_tmp41;
V53_tmp41 = V53_tmp35 * V49_i;
npy_float64 V53_tmp42;
V53_tmp42 = V39_i * V53_tmp8;
npy_float64 V53_tmp43;
V53_tmp43 = V53_tmp10 ? V25_i : V53_tmp42;
npy_float64 V53_tmp44;
V53_tmp44 = log((npy_float64)V53_tmp17);
npy_float64 V53_tmp45;
V53_tmp45 = V41_i * V53_tmp44;
npy_float64 V53_tmp46;
V53_tmp46 = V53_tmp18 ? V27_i : V53_tmp45;
npy_float64 V53_tmp47;
V53_tmp47 = V41_i + V39_i;
npy_float64 V53_tmp48;
V53_tmp48 = lgamma((npy_float64)V53_tmp47);
npy_float64 V53_tmp49;
V53_tmp49 = V53_tmp48 - V37_i;
npy_float64 V53_tmp50;
V53_tmp50 = V53_tmp49 - V21_i;
npy_float64 V53_tmp51;
V53_tmp51 = V53_tmp50 + V53_tmp46 + V53_tmp43;
npy_float64 V53_tmp52;
V53_tmp52 = (npy_float64) _psi(V53_tmp47);
npy_float64 V53_tmp53;
V53_tmp53 = V29_i ? (0.0) : V53_tmp52;
V1_i = V53_tmp4;
V3_i = V53_tmp5;
V5_i = V53_tmp11;
V7_i = V53_tmp12;
V9_i = V53_tmp22;
V11_i = V53_tmp35;
V13_i = V53_tmp40;
V15_i = V53_tmp41;
V17_i = V53_tmp51;
V19_i = V53_tmp53;
}

            #undef V5_i
#undef V7_i

        }
        
        }
        }
__label_53:

double __DUMMY_53;

}
__label_52:

        if (V51) {
            Py_XDECREF(V51);
        }
        
    {Py_XDECREF(py_V51);}
    
double __DUMMY_52;

}
__label_50:

        if (V49) {
            Py_XDECREF(V49);
        }
        
    {Py_XDECREF(py_V49);}
    
double __DUMMY_50;

}
__label_48:

        if (V47) {
            Py_XDECREF(V47);
        }
        
    {Py_XDECREF(py_V47);}
    
double __DUMMY_48;

}
__label_46:

        if (V45) {
            Py_XDECREF(V45);
        }
        
    {Py_XDECREF(py_V45);}
    
double __DUMMY_46;

}
__label_44:

        if (V43) {
            Py_XDECREF(V43);
        }
        
    {Py_XDECREF(py_V43);}
    
double __DUMMY_44;

}
__label_42:

        if (V41) {
            Py_XDECREF(V41);
        }
        
    {Py_XDECREF(py_V41);}
    
double __DUMMY_42;

}
__label_40:

        if (V39) {
            Py_XDECREF(V39);
        }
        
    {Py_XDECREF(py_V39);}
    
double __DUMMY_40;

}
__label_38:

        if (V37) {
            Py_XDECREF(V37);
        }
        
    {Py_XDECREF(py_V37);}
    
double __DUMMY_38;

}
__label_36:

        if (V35) {
            Py_XDECREF(V35);
        }
        
    {Py_XDECREF(py_V35);}
    
double __DUMMY_36;

}
__label_34:

        if (V33) {
            Py_XDECREF(V33);
        }
        
    {Py_XDECREF(py_V33);}
    
double __DUMMY_34;

}
__label_32:

        if (V31) {
            Py_XDECREF(V31);
        }
        
    {Py_XDECREF(py_V31);}
    
double __DUMMY_32;

}
__label_30:

        if (V29) {
            Py_XDECREF(V29);
        }
        
    {Py_XDECREF(py_V29);}
    
double __DUMMY_30;

}
__label_28:

        if (V27) {
            Py_XDECREF(V27);
        }
        
    {Py_XDECREF(py_V27);}
    
double __DUMMY_28;

}
__label_26:

        if (V25) {
            Py_XDECREF(V25);
        }
        
    {Py_XDECREF(py_V25);}
    
double __DUMMY_26;

}
__label_24:

        if (V23) {
            Py_XDECREF(V23);
        }
        
    {Py_XDECREF(py_V23);}
    
double __DUMMY_24;

}
__label_22:

        if (V21) {
            Py_XDECREF(V21);
        }
        
    {Py_XDECREF(py_V21);}
    
double __DUMMY_22;

}
__label_20:

    if (!__failure) {
      
        {Py_XDECREF(py_V19);}
        if (!V19) {
            Py_INCREF(Py_None);
            py_V19 = Py_None;
        }
        else if ((void*)py_V19 != (void*)V19) {
            py_V19 = (PyObject*)V19;
        }

        {Py_XINCREF(py_V19);}

        if (V19 && !PyArray_ISALIGNED((PyArrayObject*) py_V19)) {
            PyErr_Format(PyExc_NotImplementedError,
                         "c_sync: expected an aligned array, got non-aligned array of type %ld"
                         " with %ld dimensions, with 3 last dims "
                         "%ld, %ld, %ld"
                         " and 3 last strides %ld %ld, %ld.",
                         (long int) PyArray_TYPE((PyArrayObject*) py_V19),
                         (long int) PyArray_NDIM(V19),
                         (long int) (PyArray_NDIM(V19) >= 3 ?
        PyArray_DIMS(V19)[PyArray_NDIM(V19)-3] : -1),
                         (long int) (PyArray_NDIM(V19) >= 2 ?
        PyArray_DIMS(V19)[PyArray_NDIM(V19)-2] : -1),
                         (long int) (PyArray_NDIM(V19) >= 1 ?
        PyArray_DIMS(V19)[PyArray_NDIM(V19)-1] : -1),
                         (long int) (PyArray_NDIM(V19) >= 3 ?
        PyArray_STRIDES(V19)[PyArray_NDIM(V19)-3] : -1),
                         (long int) (PyArray_NDIM(V19) >= 2 ?
        PyArray_STRIDES(V19)[PyArray_NDIM(V19)-2] : -1),
                         (long int) (PyArray_NDIM(V19) >= 1 ?
        PyArray_STRIDES(V19)[PyArray_NDIM(V19)-1] : -1)
        );
            {
        __failure = 20;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_20;}
        }
        
      PyObject* old = PyList_GET_ITEM(storage_V19, 0);
      {Py_XINCREF(py_V19);}
      PyList_SET_ITEM(storage_V19, 0, py_V19);
      {Py_XDECREF(old);}
    }
    
        if (V19) {
            Py_XDECREF(V19);
        }
        
    {Py_XDECREF(py_V19);}
    
double __DUMMY_20;

}
__label_18:

    if (!__failure) {
      
        {Py_XDECREF(py_V17);}
        if (!V17) {
            Py_INCREF(Py_None);
            py_V17 = Py_None;
        }
        else if ((void*)py_V17 != (void*)V17) {
            py_V17 = (PyObject*)V17;
        }

        {Py_XINCREF(py_V17);}

        if (V17 && !PyArray_ISALIGNED((PyArrayObject*) py_V17)) {
            PyErr_Format(PyExc_NotImplementedError,
                         "c_sync: expected an aligned array, got non-aligned array of type %ld"
                         " with %ld dimensions, with 3 last dims "
                         "%ld, %ld, %ld"
                         " and 3 last strides %ld %ld, %ld.",
                         (long int) PyArray_TYPE((PyArrayObject*) py_V17),
                         (long int) PyArray_NDIM(V17),
                         (long int) (PyArray_NDIM(V17) >= 3 ?
        PyArray_DIMS(V17)[PyArray_NDIM(V17)-3] : -1),
                         (long int) (PyArray_NDIM(V17) >= 2 ?
        PyArray_DIMS(V17)[PyArray_NDIM(V17)-2] : -1),
                         (long int) (PyArray_NDIM(V17) >= 1 ?
        PyArray_DIMS(V17)[PyArray_NDIM(V17)-1] : -1),
                         (long int) (PyArray_NDIM(V17) >= 3 ?
        PyArray_STRIDES(V17)[PyArray_NDIM(V17)-3] : -1),
                         (long int) (PyArray_NDIM(V17) >= 2 ?
        PyArray_STRIDES(V17)[PyArray_NDIM(V17)-2] : -1),
                         (long int) (PyArray_NDIM(V17) >= 1 ?
        PyArray_STRIDES(V17)[PyArray_NDIM(V17)-1] : -1)
        );
            {
        __failure = 18;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_18;}
        }
        
      PyObject* old = PyList_GET_ITEM(storage_V17, 0);
      {Py_XINCREF(py_V17);}
      PyList_SET_ITEM(storage_V17, 0, py_V17);
      {Py_XDECREF(old);}
    }
    
        if (V17) {
            Py_XDECREF(V17);
        }
        
    {Py_XDECREF(py_V17);}
    
double __DUMMY_18;

}
__label_16:

    if (!__failure) {
      
        {Py_XDECREF(py_V15);}
        if (!V15) {
            Py_INCREF(Py_None);
            py_V15 = Py_None;
        }
        else if ((void*)py_V15 != (void*)V15) {
            py_V15 = (PyObject*)V15;
        }

        {Py_XINCREF(py_V15);}

        if (V15 && !PyArray_ISALIGNED((PyArrayObject*) py_V15)) {
            PyErr_Format(PyExc_NotImplementedError,
                         "c_sync: expected an aligned array, got non-aligned array of type %ld"
                         " with %ld dimensions, with 3 last dims "
                         "%ld, %ld, %ld"
                         " and 3 last strides %ld %ld, %ld.",
                         (long int) PyArray_TYPE((PyArrayObject*) py_V15),
                         (long int) PyArray_NDIM(V15),
                         (long int) (PyArray_NDIM(V15) >= 3 ?
        PyArray_DIMS(V15)[PyArray_NDIM(V15)-3] : -1),
                         (long int) (PyArray_NDIM(V15) >= 2 ?
        PyArray_DIMS(V15)[PyArray_NDIM(V15)-2] : -1),
                         (long int) (PyArray_NDIM(V15) >= 1 ?
        PyArray_DIMS(V15)[PyArray_NDIM(V15)-1] : -1),
                         (long int) (PyArray_NDIM(V15) >= 3 ?
        PyArray_STRIDES(V15)[PyArray_NDIM(V15)-3] : -1),
                         (long int) (PyArray_NDIM(V15) >= 2 ?
        PyArray_STRIDES(V15)[PyArray_NDIM(V15)-2] : -1),
                         (long int) (PyArray_NDIM(V15) >= 1 ?
        PyArray_STRIDES(V15)[PyArray_NDIM(V15)-1] : -1)
        );
            {
        __failure = 16;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_16;}
        }
        
      PyObject* old = PyList_GET_ITEM(storage_V15, 0);
      {Py_XINCREF(py_V15);}
      PyList_SET_ITEM(storage_V15, 0, py_V15);
      {Py_XDECREF(old);}
    }
    
        if (V15) {
            Py_XDECREF(V15);
        }
        
    {Py_XDECREF(py_V15);}
    
double __DUMMY_16;

}
__label_14:

    if (!__failure) {
      
        {Py_XDECREF(py_V13);}
        if (!V13) {
            Py_INCREF(Py_None);
            py_V13 = Py_None;
        }
        else if ((void*)py_V13 != (void*)V13) {
            py_V13 = (PyObject*)V13;
        }

        {Py_XINCREF(py_V13);}

        if (V13 && !PyArray_ISALIGNED((PyArrayObject*) py_V13)) {
            PyErr_Format(PyExc_NotImplementedError,
                         "c_sync: expected an aligned array, got non-aligned array of type %ld"
                         " with %ld dimensions, with 3 last dims "
                         "%ld, %ld, %ld"
                         " and 3 last strides %ld %ld, %ld.",
                         (long int) PyArray_TYPE((PyArrayObject*) py_V13),
                         (long int) PyArray_NDIM(V13),
                         (long int) (PyArray_NDIM(V13) >= 3 ?
        PyArray_DIMS(V13)[PyArray_NDIM(V13)-3] : -1),
                         (long int) (PyArray_NDIM(V13) >= 2 ?
        PyArray_DIMS(V13)[PyArray_NDIM(V13)-2] : -1),
                         (long int) (PyArray_NDIM(V13) >= 1 ?
        PyArray_DIMS(V13)[PyArray_NDIM(V13)-1] : -1),
                         (long int) (PyArray_NDIM(V13) >= 3 ?
        PyArray_STRIDES(V13)[PyArray_NDIM(V13)-3] : -1),
                         (long int) (PyArray_NDIM(V13) >= 2 ?
        PyArray_STRIDES(V13)[PyArray_NDIM(V13)-2] : -1),
                         (long int) (PyArray_NDIM(V13) >= 1 ?
        PyArray_STRIDES(V13)[PyArray_NDIM(V13)-1] : -1)
        );
            {
        __failure = 14;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_14;}
        }
        
      PyObject* old = PyList_GET_ITEM(storage_V13, 0);
      {Py_XINCREF(py_V13);}
      PyList_SET_ITEM(storage_V13, 0, py_V13);
      {Py_XDECREF(old);}
    }
    
        if (V13) {
            Py_XDECREF(V13);
        }
        
    {Py_XDECREF(py_V13);}
    
double __DUMMY_14;

}
__label_12:

    if (!__failure) {
      
        {Py_XDECREF(py_V11);}
        if (!V11) {
            Py_INCREF(Py_None);
            py_V11 = Py_None;
        }
        else if ((void*)py_V11 != (void*)V11) {
            py_V11 = (PyObject*)V11;
        }

        {Py_XINCREF(py_V11);}

        if (V11 && !PyArray_ISALIGNED((PyArrayObject*) py_V11)) {
            PyErr_Format(PyExc_NotImplementedError,
                         "c_sync: expected an aligned array, got non-aligned array of type %ld"
                         " with %ld dimensions, with 3 last dims "
                         "%ld, %ld, %ld"
                         " and 3 last strides %ld %ld, %ld.",
                         (long int) PyArray_TYPE((PyArrayObject*) py_V11),
                         (long int) PyArray_NDIM(V11),
                         (long int) (PyArray_NDIM(V11) >= 3 ?
        PyArray_DIMS(V11)[PyArray_NDIM(V11)-3] : -1),
                         (long int) (PyArray_NDIM(V11) >= 2 ?
        PyArray_DIMS(V11)[PyArray_NDIM(V11)-2] : -1),
                         (long int) (PyArray_NDIM(V11) >= 1 ?
        PyArray_DIMS(V11)[PyArray_NDIM(V11)-1] : -1),
                         (long int) (PyArray_NDIM(V11) >= 3 ?
        PyArray_STRIDES(V11)[PyArray_NDIM(V11)-3] : -1),
                         (long int) (PyArray_NDIM(V11) >= 2 ?
        PyArray_STRIDES(V11)[PyArray_NDIM(V11)-2] : -1),
                         (long int) (PyArray_NDIM(V11) >= 1 ?
        PyArray_STRIDES(V11)[PyArray_NDIM(V11)-1] : -1)
        );
            {
        __failure = 12;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_12;}
        }
        
      PyObject* old = PyList_GET_ITEM(storage_V11, 0);
      {Py_XINCREF(py_V11);}
      PyList_SET_ITEM(storage_V11, 0, py_V11);
      {Py_XDECREF(old);}
    }
    
        if (V11) {
            Py_XDECREF(V11);
        }
        
    {Py_XDECREF(py_V11);}
    
double __DUMMY_12;

}
__label_10:

    if (!__failure) {
      
        {Py_XDECREF(py_V9);}
        if (!V9) {
            Py_INCREF(Py_None);
            py_V9 = Py_None;
        }
        else if ((void*)py_V9 != (void*)V9) {
            py_V9 = (PyObject*)V9;
        }

        {Py_XINCREF(py_V9);}

        if (V9 && !PyArray_ISALIGNED((PyArrayObject*) py_V9)) {
            PyErr_Format(PyExc_NotImplementedError,
                         "c_sync: expected an aligned array, got non-aligned array of type %ld"
                         " with %ld dimensions, with 3 last dims "
                         "%ld, %ld, %ld"
                         " and 3 last strides %ld %ld, %ld.",
                         (long int) PyArray_TYPE((PyArrayObject*) py_V9),
                         (long int) PyArray_NDIM(V9),
                         (long int) (PyArray_NDIM(V9) >= 3 ?
        PyArray_DIMS(V9)[PyArray_NDIM(V9)-3] : -1),
                         (long int) (PyArray_NDIM(V9) >= 2 ?
        PyArray_DIMS(V9)[PyArray_NDIM(V9)-2] : -1),
                         (long int) (PyArray_NDIM(V9) >= 1 ?
        PyArray_DIMS(V9)[PyArray_NDIM(V9)-1] : -1),
                         (long int) (PyArray_NDIM(V9) >= 3 ?
        PyArray_STRIDES(V9)[PyArray_NDIM(V9)-3] : -1),
                         (long int) (PyArray_NDIM(V9) >= 2 ?
        PyArray_STRIDES(V9)[PyArray_NDIM(V9)-2] : -1),
                         (long int) (PyArray_NDIM(V9) >= 1 ?
        PyArray_STRIDES(V9)[PyArray_NDIM(V9)-1] : -1)
        );
            {
        __failure = 10;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_10;}
        }
        
      PyObject* old = PyList_GET_ITEM(storage_V9, 0);
      {Py_XINCREF(py_V9);}
      PyList_SET_ITEM(storage_V9, 0, py_V9);
      {Py_XDECREF(old);}
    }
    
        if (V9) {
            Py_XDECREF(V9);
        }
        
    {Py_XDECREF(py_V9);}
    
double __DUMMY_10;

}
__label_8:

    if (!__failure) {
      
        {Py_XDECREF(py_V7);}
        if (!V7) {
            Py_INCREF(Py_None);
            py_V7 = Py_None;
        }
        else if ((void*)py_V7 != (void*)V7) {
            py_V7 = (PyObject*)V7;
        }

        {Py_XINCREF(py_V7);}

        if (V7 && !PyArray_ISALIGNED((PyArrayObject*) py_V7)) {
            PyErr_Format(PyExc_NotImplementedError,
                         "c_sync: expected an aligned array, got non-aligned array of type %ld"
                         " with %ld dimensions, with 3 last dims "
                         "%ld, %ld, %ld"
                         " and 3 last strides %ld %ld, %ld.",
                         (long int) PyArray_TYPE((PyArrayObject*) py_V7),
                         (long int) PyArray_NDIM(V7),
                         (long int) (PyArray_NDIM(V7) >= 3 ?
        PyArray_DIMS(V7)[PyArray_NDIM(V7)-3] : -1),
                         (long int) (PyArray_NDIM(V7) >= 2 ?
        PyArray_DIMS(V7)[PyArray_NDIM(V7)-2] : -1),
                         (long int) (PyArray_NDIM(V7) >= 1 ?
        PyArray_DIMS(V7)[PyArray_NDIM(V7)-1] : -1),
                         (long int) (PyArray_NDIM(V7) >= 3 ?
        PyArray_STRIDES(V7)[PyArray_NDIM(V7)-3] : -1),
                         (long int) (PyArray_NDIM(V7) >= 2 ?
        PyArray_STRIDES(V7)[PyArray_NDIM(V7)-2] : -1),
                         (long int) (PyArray_NDIM(V7) >= 1 ?
        PyArray_STRIDES(V7)[PyArray_NDIM(V7)-1] : -1)
        );
            {
        __failure = 8;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_8;}
        }
        
      PyObject* old = PyList_GET_ITEM(storage_V7, 0);
      {Py_XINCREF(py_V7);}
      PyList_SET_ITEM(storage_V7, 0, py_V7);
      {Py_XDECREF(old);}
    }
    
        if (V7) {
            Py_XDECREF(V7);
        }
        
    {Py_XDECREF(py_V7);}
    
double __DUMMY_8;

}
__label_6:

    if (!__failure) {
      
        {Py_XDECREF(py_V5);}
        if (!V5) {
            Py_INCREF(Py_None);
            py_V5 = Py_None;
        }
        else if ((void*)py_V5 != (void*)V5) {
            py_V5 = (PyObject*)V5;
        }

        {Py_XINCREF(py_V5);}

        if (V5 && !PyArray_ISALIGNED((PyArrayObject*) py_V5)) {
            PyErr_Format(PyExc_NotImplementedError,
                         "c_sync: expected an aligned array, got non-aligned array of type %ld"
                         " with %ld dimensions, with 3 last dims "
                         "%ld, %ld, %ld"
                         " and 3 last strides %ld %ld, %ld.",
                         (long int) PyArray_TYPE((PyArrayObject*) py_V5),
                         (long int) PyArray_NDIM(V5),
                         (long int) (PyArray_NDIM(V5) >= 3 ?
        PyArray_DIMS(V5)[PyArray_NDIM(V5)-3] : -1),
                         (long int) (PyArray_NDIM(V5) >= 2 ?
        PyArray_DIMS(V5)[PyArray_NDIM(V5)-2] : -1),
                         (long int) (PyArray_NDIM(V5) >= 1 ?
        PyArray_DIMS(V5)[PyArray_NDIM(V5)-1] : -1),
                         (long int) (PyArray_NDIM(V5) >= 3 ?
        PyArray_STRIDES(V5)[PyArray_NDIM(V5)-3] : -1),
                         (long int) (PyArray_NDIM(V5) >= 2 ?
        PyArray_STRIDES(V5)[PyArray_NDIM(V5)-2] : -1),
                         (long int) (PyArray_NDIM(V5) >= 1 ?
        PyArray_STRIDES(V5)[PyArray_NDIM(V5)-1] : -1)
        );
            {
        __failure = 6;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_6;}
        }
        
      PyObject* old = PyList_GET_ITEM(storage_V5, 0);
      {Py_XINCREF(py_V5);}
      PyList_SET_ITEM(storage_V5, 0, py_V5);
      {Py_XDECREF(old);}
    }
    
        if (V5) {
            Py_XDECREF(V5);
        }
        
    {Py_XDECREF(py_V5);}
    
double __DUMMY_6;

}
__label_4:

    if (!__failure) {
      
        {Py_XDECREF(py_V3);}
        if (!V3) {
            Py_INCREF(Py_None);
            py_V3 = Py_None;
        }
        else if ((void*)py_V3 != (void*)V3) {
            py_V3 = (PyObject*)V3;
        }

        {Py_XINCREF(py_V3);}

        if (V3 && !PyArray_ISALIGNED((PyArrayObject*) py_V3)) {
            PyErr_Format(PyExc_NotImplementedError,
                         "c_sync: expected an aligned array, got non-aligned array of type %ld"
                         " with %ld dimensions, with 3 last dims "
                         "%ld, %ld, %ld"
                         " and 3 last strides %ld %ld, %ld.",
                         (long int) PyArray_TYPE((PyArrayObject*) py_V3),
                         (long int) PyArray_NDIM(V3),
                         (long int) (PyArray_NDIM(V3) >= 3 ?
        PyArray_DIMS(V3)[PyArray_NDIM(V3)-3] : -1),
                         (long int) (PyArray_NDIM(V3) >= 2 ?
        PyArray_DIMS(V3)[PyArray_NDIM(V3)-2] : -1),
                         (long int) (PyArray_NDIM(V3) >= 1 ?
        PyArray_DIMS(V3)[PyArray_NDIM(V3)-1] : -1),
                         (long int) (PyArray_NDIM(V3) >= 3 ?
        PyArray_STRIDES(V3)[PyArray_NDIM(V3)-3] : -1),
                         (long int) (PyArray_NDIM(V3) >= 2 ?
        PyArray_STRIDES(V3)[PyArray_NDIM(V3)-2] : -1),
                         (long int) (PyArray_NDIM(V3) >= 1 ?
        PyArray_STRIDES(V3)[PyArray_NDIM(V3)-1] : -1)
        );
            {
        __failure = 4;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_4;}
        }
        
      PyObject* old = PyList_GET_ITEM(storage_V3, 0);
      {Py_XINCREF(py_V3);}
      PyList_SET_ITEM(storage_V3, 0, py_V3);
      {Py_XDECREF(old);}
    }
    
        if (V3) {
            Py_XDECREF(V3);
        }
        
    {Py_XDECREF(py_V3);}
    
double __DUMMY_4;

}
__label_2:

    if (!__failure) {
      
        {Py_XDECREF(py_V1);}
        if (!V1) {
            Py_INCREF(Py_None);
            py_V1 = Py_None;
        }
        else if ((void*)py_V1 != (void*)V1) {
            py_V1 = (PyObject*)V1;
        }

        {Py_XINCREF(py_V1);}

        if (V1 && !PyArray_ISALIGNED((PyArrayObject*) py_V1)) {
            PyErr_Format(PyExc_NotImplementedError,
                         "c_sync: expected an aligned array, got non-aligned array of type %ld"
                         " with %ld dimensions, with 3 last dims "
                         "%ld, %ld, %ld"
                         " and 3 last strides %ld %ld, %ld.",
                         (long int) PyArray_TYPE((PyArrayObject*) py_V1),
                         (long int) PyArray_NDIM(V1),
                         (long int) (PyArray_NDIM(V1) >= 3 ?
        PyArray_DIMS(V1)[PyArray_NDIM(V1)-3] : -1),
                         (long int) (PyArray_NDIM(V1) >= 2 ?
        PyArray_DIMS(V1)[PyArray_NDIM(V1)-2] : -1),
                         (long int) (PyArray_NDIM(V1) >= 1 ?
        PyArray_DIMS(V1)[PyArray_NDIM(V1)-1] : -1),
                         (long int) (PyArray_NDIM(V1) >= 3 ?
        PyArray_STRIDES(V1)[PyArray_NDIM(V1)-3] : -1),
                         (long int) (PyArray_NDIM(V1) >= 2 ?
        PyArray_STRIDES(V1)[PyArray_NDIM(V1)-2] : -1),
                         (long int) (PyArray_NDIM(V1) >= 1 ?
        PyArray_STRIDES(V1)[PyArray_NDIM(V1)-1] : -1)
        );
            {
        __failure = 2;
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                "Unexpected error in an Op's C code. "
                "No Python exception was set.");
        }
        goto __label_2;}
        }
        
      PyObject* old = PyList_GET_ITEM(storage_V1, 0);
      {Py_XINCREF(py_V1);}
      PyList_SET_ITEM(storage_V1, 0, py_V1);
      {Py_XDECREF(old);}
    }
    
        if (V1) {
            Py_XDECREF(V1);
        }
        
    {Py_XDECREF(py_V1);}
    
double __DUMMY_2;

}

            
        if (__failure) {
            // When there is a failure, this code puts the exception
            // in __ERROR.
            PyObject* err_type = NULL;
            PyObject* err_msg = NULL;
            PyObject* err_traceback = NULL;
            PyErr_Fetch(&err_type, &err_msg, &err_traceback);
            if (!err_type) {err_type = Py_None;Py_INCREF(Py_None);}
            if (!err_msg) {err_msg = Py_None; Py_INCREF(Py_None);}
            if (!err_traceback) {err_traceback = Py_None; Py_INCREF(Py_None);}
            PyObject* old_err_type = PyList_GET_ITEM(__ERROR, 0);
            PyObject* old_err_msg = PyList_GET_ITEM(__ERROR, 1);
            PyObject* old_err_traceback = PyList_GET_ITEM(__ERROR, 2);
            PyList_SET_ITEM(__ERROR, 0, err_type);
            PyList_SET_ITEM(__ERROR, 1, err_msg);
            PyList_SET_ITEM(__ERROR, 2, err_traceback);
            {Py_XDECREF(old_err_type);}
            {Py_XDECREF(old_err_msg);}
            {Py_XDECREF(old_err_traceback);}
        }
        // The failure code is returned to index what code block failed.
        return __failure;
        
        }
    };
    }
    

        static int __struct_compiled_op_mbd9a99ccae6f8313aa7113832d471284ef2a555e009e6ae103d62f0b8be82a12_executor(__struct_compiled_op_mbd9a99ccae6f8313aa7113832d471284ef2a555e009e6ae103d62f0b8be82a12 *self) {
            return self->run();
        }

        static void __struct_compiled_op_mbd9a99ccae6f8313aa7113832d471284ef2a555e009e6ae103d62f0b8be82a12_destructor(PyObject *capsule) {
            __struct_compiled_op_mbd9a99ccae6f8313aa7113832d471284ef2a555e009e6ae103d62f0b8be82a12 *self = (__struct_compiled_op_mbd9a99ccae6f8313aa7113832d471284ef2a555e009e6ae103d62f0b8be82a12 *)PyCapsule_GetContext(capsule);
            delete self;
        }
    
//////////////////////
////  Functions
//////////////////////
static PyObject * instantiate(PyObject * self, PyObject *argtuple) {
  assert(PyTuple_Check(argtuple));
  if (27 != PyTuple_Size(argtuple)){ 
     PyErr_Format(PyExc_TypeError, "Wrong number of arguments, expected 27, got %%i", (int)PyTuple_Size(argtuple));
     return NULL;
  }
  __struct_compiled_op_mbd9a99ccae6f8313aa7113832d471284ef2a555e009e6ae103d62f0b8be82a12* struct_ptr = new __struct_compiled_op_mbd9a99ccae6f8313aa7113832d471284ef2a555e009e6ae103d62f0b8be82a12();
  if (struct_ptr->init( PyTuple_GET_ITEM(argtuple, 0),PyTuple_GET_ITEM(argtuple, 1),PyTuple_GET_ITEM(argtuple, 2),PyTuple_GET_ITEM(argtuple, 3),PyTuple_GET_ITEM(argtuple, 4),PyTuple_GET_ITEM(argtuple, 5),PyTuple_GET_ITEM(argtuple, 6),PyTuple_GET_ITEM(argtuple, 7),PyTuple_GET_ITEM(argtuple, 8),PyTuple_GET_ITEM(argtuple, 9),PyTuple_GET_ITEM(argtuple, 10),PyTuple_GET_ITEM(argtuple, 11),PyTuple_GET_ITEM(argtuple, 12),PyTuple_GET_ITEM(argtuple, 13),PyTuple_GET_ITEM(argtuple, 14),PyTuple_GET_ITEM(argtuple, 15),PyTuple_GET_ITEM(argtuple, 16),PyTuple_GET_ITEM(argtuple, 17),PyTuple_GET_ITEM(argtuple, 18),PyTuple_GET_ITEM(argtuple, 19),PyTuple_GET_ITEM(argtuple, 20),PyTuple_GET_ITEM(argtuple, 21),PyTuple_GET_ITEM(argtuple, 22),PyTuple_GET_ITEM(argtuple, 23),PyTuple_GET_ITEM(argtuple, 24),PyTuple_GET_ITEM(argtuple, 25),PyTuple_GET_ITEM(argtuple, 26) ) != 0) {
    delete struct_ptr;
    return NULL;
  }
    PyObject* thunk = PyCapsule_New((void*)(&__struct_compiled_op_mbd9a99ccae6f8313aa7113832d471284ef2a555e009e6ae103d62f0b8be82a12_executor), NULL, __struct_compiled_op_mbd9a99ccae6f8313aa7113832d471284ef2a555e009e6ae103d62f0b8be82a12_destructor);
    if (thunk != NULL && PyCapsule_SetContext(thunk, struct_ptr) != 0) {
        PyErr_Clear();
        Py_DECREF(thunk);
        thunk = NULL;
    }

  return thunk; }

//////////////////////
////  Module init
//////////////////////
static PyMethodDef MyMethods[] = {
	{"instantiate", instantiate, METH_VARARGS, "undocumented"} ,
	{NULL, NULL, 0, NULL}
};
static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "mbd9a99ccae6f8313aa7113832d471284ef2a555e009e6ae103d62f0b8be82a12",
  NULL,
  -1,
  MyMethods,
};

PyMODINIT_FUNC PyInit_mbd9a99ccae6f8313aa7113832d471284ef2a555e009e6ae103d62f0b8be82a12(void) {
   import_array();
   
    PyObject *m = PyModule_Create(&moduledef);
    return m;
}
