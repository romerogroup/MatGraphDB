MatGraphDB Docs!
========================================

**MatGraphDB** is a Python package designed to simplify graph-based data management and analysis in materials and molecular science. It enables researchers to efficiently transform complex theoretical data into structured graph representations, leveraging:

- **High-performance storage:** Utilizes Apache Parquet for scalable and rapid data access.
- **Automated workflows:** Converts theoretical and computational data into graph structures seamlessly.
- **Robust data operations:** Offers comprehensive CRUD functionality and custom generators to maintain consistent relationships between entities.

By streamlining these processes, MatGraphDB makes advanced graph analysis more accessible, helping users model and predict material properties with ease.


+++++++++++++++
Installation
+++++++++++++++

If you're new to **MatGraphDB**, you're in the right place!

.. code-block:: bash

   pip install matgraphdb

.. +++++++++++++++
.. Paper
.. +++++++++++++++

.. .. grid:: 2

..    .. grid-item-card:: 1st Publication
..       :link-type: ref
..       :class-title: pyprocar-card-title

..       Using ParquetDB in your research? Please consider citing or acknowledging
..       us.  We have a Arxiv Publication!

..       .. image:: media/images/arxiv_publication.png
..          :target: https://arxiv.org/pdf/2502.05311




+++++++++++++++
What Next?
+++++++++++++++

Now that you have successfully installed MatGraphDB, here are some recommended next steps:

- **Tutorials**  
  Visit the :doc:`01_tutorials/index` section for a hands-on tutorial covering the basics of creating, reading, and querying MatGraphDB files.


- **Learn the Inner Details**  
  Visit the :doc:`MatGraphDB Internals <02_internal/index>` section to dive deeper into MatGraphDB's internals to understand how it wraps PyArrow, processes different data types, and performs efficient read/write operations.

- **Example Gallery**  
  Visit the :doc:`Example Gallery <examples/index>` section real use cases of MatGraphDB.


- **Explore PyArrow**  
  MatGraphDB relies on PyArrow for powerful data type handling and filtering mechanisms. For more in-depth information on PyArrow's `Table <https://arrow.apache.org/docs/python/generated/pyarrow.Table.html>`__ structure, filtering, and other features, refer to the `PyArrow Documentation <https://arrow.apache.org/docs/python/getstarted.html>`__.


Citing MatGraphDB
==================


To be added.

.. If you use MatGraphDB in your work, please cite the following paper:

.. .. code-block:: bibtex

..   @misc{lang2025parquetdblightweightpythonparquetbased,
..       title={ParquetDB: A Lightweight Python Parquet-Based Database}, 
..       author={Logan Lang and Eduardo Hernandez and Kamal Choudhary and Aldo H. Romero},
..       year={2025},
..       eprint={2502.05311},
..       archivePrefix={arXiv},
..       primaryClass={cs.DB},
..       url={https://arxiv.org/abs/2502.05311}}



Index
==================

.. toctree::
    :maxdepth: 2
    :glob:


    01_tutorials/index
    02_internal/index
    examples/index.rst
    03_api/index
    CONTRIBUTING
