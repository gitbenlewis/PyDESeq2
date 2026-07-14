pydeseq2.ds.DeseqStats
======================

.. currentmodule:: pydeseq2.ds

.. note::

   Wald tests remain the default. For a classical likelihood-ratio test, pass
   ``test="LRT"`` and a nested ``reduced`` design to
   :class:`DeseqStats`. The full design is taken from its fitted
   :class:`~pydeseq2.dds.DeseqDataSet`. A ``contrast`` is still required,
   but it only selects the reported log fold change and standard error; the LRT
   statistic and p-value are omnibus comparisons of the full and reduced models.

   The same LRT can be prepared during model fitting by passing ``test="LRT"``
   and ``reduced=...`` to
   :meth:`pydeseq2.dds.DeseqDataSet.deseq2`. With ``test=None``,
   :class:`DeseqStats` inherits that prepared test and reuses its cached results.
   Test names are case-sensitive. Only the classical chi-square LRT is supported;
   quasi-likelihood variants are outside the current scope. LRTs reject a
   non-zero ``lfc_null``, any ``alt_hypothesis``, or a supplied
   ``prior_LFC_var``.

   When Cook's outlier refitting replaces counts, the effective count matrix is
   retained in the internally managed ``dds.layers["replace_counts"]`` layer so
   both LRT models use the same data. This makes ``"replace_counts"`` a reserved
   layer name for that fit: an existing user-owned layer with that name causes
   replacement to raise instead of being overwritten. Because the full replacement
   matrix is retained, ``low_memory=True`` does not avoid this extra memory cost.

.. autoclass:: DeseqStats

   .. rubric:: Methods

   .. autosummary::

      ~DeseqStats.lfc_shrink
      ~DeseqStats.run_lrt_test
      ~DeseqStats.run_wald_test
      ~DeseqStats.summary
