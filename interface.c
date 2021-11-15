// https://www.python.org/dev/peps/pep-0353/ The conversion codes 's#'
// and 't#' will output Py_ssize_t if the macro PY_SSIZE_T_CLEAN is
// defined before Python.h is included, and continue to output int if
// that macro isn't defined. TDH 15 Nov 2021: why is this here if we
// do not use s# and t# converters?
#define PY_SSIZE_T_CLEAN 

// Declare version of numpy that we are using, otherwise there will be
// a compiler warning. https://numpy.org/doc/stable/numpy-ref.pdf
#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include "modelSelectionFwd.h"

static PyObject *
ModelSelectionInterface(PyObject *self, PyObject *args){
  PyArrayObject *loss_obj, *complexity_obj,
    *index_obj, *penalty_obj, *iterations_obj;
  if(!PyArg_ParseTuple
     (args, "O!O!",
      &PyArray_Type, &loss_obj,
      &PyArray_Type, &complexity_obj)){
    return NULL;
  }
  if(PyArray_TYPE(loss_obj)!=NPY_DOUBLE){
    PyErr_SetString
      (PyExc_TypeError,
       "loss_obj must be numpy.ndarray type float64");
    return NULL;
  }
  if(PyArray_TYPE(complexity_obj)!=NPY_DOUBLE){
    PyErr_SetString
      (PyExc_TypeError,
       "complexity_obj must be numpy.ndarray type float64");
    return NULL;
  }
  npy_intp npy_models = PyArray_DIM(loss_obj, 0);
  int n_models = npy_models;
  // outputs.
  index_obj = (PyArrayObject*)PyArray_ZEROS(1, &npy_models, NPY_INT, 0);
  penalty_obj = (PyArrayObject*)PyArray_ZEROS(1, &npy_models, NPY_DOUBLE, 0);
  iterations_obj = (PyArrayObject*)PyArray_ZEROS(1, &npy_models, NPY_INT, 0);
  int status = modelSelectionFwd
    (PyArray_DATA(loss_obj),
     PyArray_DATA(complexity_obj),
     &n_models,
     PyArray_DATA(index_obj),
     PyArray_DATA(penalty_obj),
     PyArray_DATA(iterations_obj));
  if(status == ERROR_FWD_LOSS_NOT_DECREASING){
    PyErr_SetString
      (PyExc_ValueError,
       "loss not decreasing");
    return NULL;
  }
  if(status == ERROR_FWD_COMPLEXITY_NOT_INCREASING){
    PyErr_SetString
      (PyExc_ValueError,
       "complexity not increasing");
    return NULL;
  }
  return Py_BuildValue
    ("{s:N,s:N,s:N,i}",
     "index", index_obj,
     "penalty", penalty_obj,
     "iterations", iterations_obj,
     "n_models", n_models);
}

static PyMethodDef Methods[] = {
  {"interface", ModelSelectionInterface, METH_VARARGS,
   "Exact Breakpoints in Model Selection Function"},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduleDef =
  {
    PyModuleDef_HEAD_INIT,
    "ModelSelectionInterface",
    "A Python extension for ModelSelection",
    -1,
    Methods
  };


PyMODINIT_FUNC
PyInit_ModelSelectionInterface(void)
{
  PyObject *module;
  module = PyModule_Create(&moduleDef);
  if(module == NULL) return NULL;
  import_array();//necessary from numpy otherwise we crash with segfault
  return module;
}

