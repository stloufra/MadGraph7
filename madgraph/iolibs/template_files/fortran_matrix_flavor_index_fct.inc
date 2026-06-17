      INTEGER FUNCTION %(func_name)s(FLAVOR)
C     Resolve an external FLAVOR(NEXTERNAL) group-position vector to its
C     1-based index in the allowed-flavor table (the same ordering used by
C     compute_flavor_masks / the FLAV_TABLE mask columns). The result is always
C     in [1,NFLAV]: an unresolved flavor (not in the table, not expected in
C     normal use) falls back to 1, the first allowed flavor, so callers can
C     index the 1..NFLAV GOODHEL/NTRY arrays and FLAV_TABLE unconditionally
C     (no reserved 0 slot). Computed once per phase-space point and then
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
C     Default to the first allowed flavor for an unresolved input (see above).
      %(func_name)s = 1
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
