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
   :class:`DeseqStats` inherits that prepared test and reuses its cached results
   unless a distinct inference object is supplied. Test names are case-sensitive.
   Only the classical chi-square LRT is supported;
   quasi-likelihood variants are outside the current scope. LRTs reject a
   non-zero ``lfc_null``, any ``alt_hypothesis``, or a supplied
   ``prior_LFC_var``.

   When Cook's outlier refitting replaces counts, the full and reduced LRT
   models use the same adjusted counts.

.. autoclass:: DeseqStats

   .. rubric:: Methods

   .. autosummary::

      ~DeseqStats.lfc_shrink
      ~DeseqStats.run_lrt_test
      ~DeseqStats.run_wald_test
      ~DeseqStats.summary
