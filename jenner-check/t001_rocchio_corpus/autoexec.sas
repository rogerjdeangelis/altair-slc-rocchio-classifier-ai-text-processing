options obs=100;                 /* cap input rows for the captured run */

/* The upstream script assigns the persistent library WORKX to a local
   Windows path (d:\wpswrkx). For a self-contained run we point WORKX at
   the session WORK library so the training/testing corpus builds with no
   external path. The DATA steps below are the author's, unchanged. */
libname workx (work);
