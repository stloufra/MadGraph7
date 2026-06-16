      INTEGER FUNCTION %(func_name)s(FLAVOR)
C     Resolve an external FLAVOR(NEXTERNAL) group-position vector to its
C     1-based index in the allowed-flavor table (the same ordering used by
C     compute_flavor_masks / the FLAV_TABLE mask columns). Returns 0 when the
C     flavor is not in the table. Computed once per phase-space point and then
C     threaded down to MATRIX/GET_AMP and the good-helicity filter.
%(nexternal_decl)s
      INTEGER NFLAV
      PARAMETER (NFLAV=%(nflav)d)
      INTEGER FLAVOR(NEXTERNAL)
CF2PY INTENT(IN) :: FLAVOR(NEXTERNAL)
CF2PY INTENT(OUT) :: %(func_name)s
      INTEGER FI_I, FI_J
      LOGICAL FI_MATCH
      INTEGER FI_TABLE(NEXTERNAL, NFLAV)
      DATA FI_TABLE /%(flav_table_data)s/
      %(func_name)s = 0
      DO FI_I = 1, NFLAV
        FI_MATCH = .TRUE.
        DO FI_J = 1, NEXTERNAL
          IF (FLAVOR(FI_J) .NE. FI_TABLE(FI_J, FI_I)) THEN
            FI_MATCH = .FALSE.
            EXIT
          ENDIF
        ENDDO
        IF (FI_MATCH) THEN
          %(func_name)s = FI_I
          RETURN
        ENDIF
      ENDDO
      RETURN
      END
