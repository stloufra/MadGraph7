%(python_information)s
  subroutine smatrixhel(pdgs, procid, npdg, p, ALPHAS, SCALE2, nhel, ANS)
  IMPLICIT NONE
C ALPHAS is given at scale2 (SHOULD be different of 0 for loop induced, ignore for LO)  

CF2PY double precision, intent(in), dimension(0:3,npdg) :: p
CF2PY integer, intent(in), dimension(npdg) :: pdgs
CF2PY integer, intent(in):: procid
CF2PY integer, intent(in) :: npdg
CF2PY double precision, intent(out) :: ANS
CF2PY double precision, intent(in) :: ALPHAS
CF2PY double precision, intent(in) :: SCALE2
  integer pdgs(*)
  integer npdg, nhel, procid
  double precision p(*)
  double precision ANS, ALPHAS, PI,SCALE2
  include 'coupl.inc'
  
  
  if (scale2.eq.0)then
       PI = 3.141592653589793D0
       G = 2* DSQRT(ALPHAS*PI)
       CALL UPDATE_AS_PARAM()
  else
       CALL UPDATE_AS_PARAM2(scale2, ALPHAS)
  endif

%(smatrixhel)s

      return
      end
  
      SUBROUTINE INITIALISE(PATH)
C     ROUTINE FOR F2PY to read the benchmark point.
      IMPLICIT NONE
      CHARACTER*512 PATH
CF2PY INTENT(IN) :: PATH
      CALL SETPARA(PATH)  !first call to setup the paramaters
      RETURN
      END
      
      
      subroutine CHANGE_PARA(name, value)
      implicit none
CF2PY intent(in) :: name
CF2PY intent(in) :: value

      character*512 name
      double precision value
      
      %(helreset_def)s

      include '../Source/MODEL/input.inc'
      include '../Source/MODEL/coupl.inc'

      %(helreset_setup)s

      SELECT CASE (name)
         %(parameter_setup)s
         CASE DEFAULT
            write(*,*) 'no parameter matching', name, value
      END SELECT

      return
      end
      
    subroutine update_all_coup()
    implicit none
     call coup()
    return 
    end
      

    subroutine get_pdg_order(PDG, ALLPROC)
  IMPLICIT NONE
CF2PY INTEGER, intent(out) :: PDG(%(nb_me)i,%(maxpart)i)  
CF2PY INTEGER, intent(out) :: ALLPROC(%(nb_me)i)
  INTEGER PDG(%(nb_me)i,%(maxpart)i), PDGS(%(nb_me)i,%(maxpart)i)
  INTEGER ALLPROC(%(nb_me)i),PIDs(%(nb_me)i)
  DATA PDGS/ %(pdgs)s /
  DATA PIDS/ %(pids)s /
  PDG = PDGS
  ALLPROC = PIDS
  RETURN
  END 

    subroutine get_prefix(PREFIX)
  IMPLICIT NONE
CF2PY CHARACTER*20, intent(out) :: PREFIX(%(nb_me)i)
  character*20 PREFIX(%(nb_me)i),PREF(%(nb_me)i)
  DATA PREF / '%(prefix)s'/
  PREFIX = PREF
  RETURN
  END 
 


    subroutine set_fixed_extra_scale(new_value)
    implicit none
CF2PY logical, intent(in) :: new_value
    logical new_value
                logical fixed_extra_scale
            integer maxjetflavor
            double precision mue_over_ref
            double precision mue_ref_fixed
            common/model_setup_running/maxjetflavor,fixed_extra_scale,mue_over_ref,mue_ref_fixed
  
        fixed_extra_scale = new_value
        return 
        end

    subroutine set_mue_over_ref(new_value)
    implicit none
CF2PY double precision, intent(in) :: new_value
    double precision new_value
    logical fixed_extra_scale
    integer maxjetflavor
    double precision mue_over_ref
    double precision mue_ref_fixed
    common/model_setup_running/maxjetflavor,fixed_extra_scale,mue_over_ref,mue_ref_fixed
  
    mue_over_ref = new_value
        
    return 
    end

    subroutine set_mue_ref_fixed(new_value)
    implicit none
CF2PY double precision, intent(in) :: new_value
    double precision new_value
    logical fixed_extra_scale
    integer maxjetflavor
    double precision mue_over_ref
    double precision mue_ref_fixed
    common/model_setup_running/maxjetflavor,fixed_extra_scale,mue_over_ref,mue_ref_fixed
  
    mue_ref_fixed = new_value
        
    return 
    end


    subroutine set_maxjetflavor(new_value)
    implicit none
CF2PY integer, intent(in) :: new_value
    integer new_value
    logical fixed_extra_scale
    integer maxjetflavor
    double precision mue_over_ref
    double precision mue_ref_fixed
    common/model_setup_running/maxjetflavor,fixed_extra_scale,mue_over_ref,mue_ref_fixed
  
    maxjetflavor = new_value
        
    return 
    end


    subroutine set_asmz(new_value)
    implicit none
CF2PY double precision, intent(in) :: new_value
    double precision new_value
          integer nloop
      double precision asmz
      common/a_block/asmz,nloop
    asmz = new_value
    write(*,*) "asmz is set to ", new_value
        
    return 
    end

    subroutine set_nloop(new_value)
    implicit none
CF2PY integer, intent(in) :: new_value
    integer new_value
          integer nloop
      double precision asmz
      common/a_block/asmz,nloop
    nloop = new_value
     write(*,*) "nloop is set to ", new_value
        
    return 
    end

C This function takes the number of a particle (in the generate command) and returns 
C the angles needed to put this particle as a reference for the helicity basis
      SUBROUTINE REFCHOICE(P, pNUMBER, PHI, THETA)
      IMPLICIT NONE
C     
C     CONSTANT
C     
      DOUBLE PRECISION    EPSILON
      PARAMETER (EPSILON=1D-10)

CF2PY DOUBLE PRECISION, INTENT(OUT) :: PHI
CF2PY DOUBLE PRECISION, INTENT(OUT) :: THETA
CF2PY INTEGER, INTENT(IN) :: pNUMBER
CF2PY DOUBLE PRECISION, INTENT(IN) :: P(0:3, *)
C     
C     ARGUMENT
C     
      REAL*8 P(0:3, *)
      REAL*8 PREF(0:3)
      INTEGER pNUMBER
      DOUBLE PRECISION THETA, PHI

c PREF is the 4-momentum of the particle with particle number = pNUMBER
      PREF = P(:, pNUMBER)


      IF (ABS(PREF(1)/PREF(0)) < EPSILON .AND. ABS(PREF(2)/PREF(0)) 
     &< EPSILON .AND. ABS(PREF(3)/PREF(0)) < EPSILON ) THEN
          WRITE(*,*) "The chosen particle is immobile. We cant use it",
     &" as reference for the helicity basis"
          STOP "Error when passing to helicity basis"
      ENDIF

c The angles phi and theta are calculated such that after rotation, 
c pref = (E, 0, 0, p_k)
      PHI = SIGN(1D0, PREF(2)) * DACOS(PREF(1)/DSQRT(PREF(1)**2 +
     & PREF(2)**2))
      THETA = DACOS(PREF(3)/DSQRT(PREF(1)**2 + PREF(2)**2 + PREF(3)**2))

      END



C This function takes a given 4-momentum and returns the angles needed to put 
C this particle as a reference for the helicity basis
      SUBROUTINE REFCHOICEP(PREF, PHI, THETA)
      IMPLICIT NONE
C     
C     CONSTANT
C     
      DOUBLE PRECISION    EPSILON
      PARAMETER (EPSILON=1D-10)

CF2PY DOUBLE PRECISION, INTENT(OUT) :: PHI
CF2PY DOUBLE PRECISION INTENT(OUT) :: THETA
CF2PY DOUBLE PRECISION, INTENT(IN) :: PREF(0:3)
C     
C     ARGUMENT
C     
      REAL*8 PREF(0:3)
      DOUBLE PRECISION THETA, PHI

      IF (ABS(PREF(1)/PREF(0)) < EPSILON .AND. ABS(PREF(2)/PREF(0)) 
     &< EPSILON .AND. ABS(PREF(3)/PREF(0)) < EPSILON ) THEN
         WRITE(*,*) "The chosen particle is immobile. We cant use it",
     & 		    " as reference for the helicity basis"
         STOP "Error when passing to helicity basis"
      ENDIF

      PHI = SIGN(1D0, PREF(2)) * DACOS(PREF(1)/DSQRT(PREF(1)**2 +
     & PREF(2)**2))
      THETA = DACOS(PREF(3)/DSQRT(PREF(1)**2 + PREF(2)**2 + PREF(3)**2))
      
      END


c This function takes angles PHI and THETA and rotates the 4-momenta of all external particles
      SUBROUTINE ROTATIONP(P, PHI, THETA, NEXTERNAL, PROT)
      IMPLICIT NONE
C     
C     CONSTANT
C     
      DOUBLE PRECISION    EPSILON
      PARAMETER (EPSILON=1D-10)

CF2PY DOUBLE PRECISION, INTENT(OUT) :: PROT(0:3, NEXTERNAL)
CF2PY INTEGER, INTENT(IN) :: NEXTERNAL
CF2PY DOUBLE PRECISION, INTENT(IN) :: PHI
CF2PY DOUBLE PRECISION, INTENT(IN) :: THETA
CF2PY DOUBLE PRECISION, INTENT(IN) :: P(0:3, NEXTERNAL)
C     
C     ARGUMENT
C     
      REAL*8 P(0:3, *)
      REAL*8 PROT(0:3, *)
      DOUBLE PRECISION THETA, PHI
      INTEGER I, NEXTERNAL

      DO I= 1, NEXTERNAL
          PROT(0, I) = P(0, I)

          PROT(1, I) = -DSIN(PHI)*P(1, I) + DCOS(PHI)*P(2, I) !p_n
          IF (ABS(PROT(1, I)/PROT(0, I)) < EPSILON) THEN
            PROT(1, I) = 0D0
          ENDIF

          PROT(2, I) = -DCOS(PHI)*DCOS(THETA)*P(1, I) - DSIN(PHI) ! p_r
     & *DCOS(THETA)*P(2, I) + DSIN(THETA)*P(3, I)
          IF (ABS(PROT(2, I)/PROT(0, I)) < EPSILON) THEN
            PROT(2, I) = 0D0
          ENDIF
        
          PROT(3, I) = DCOS(PHI)*DSIN(THETA)*P(1, I) + DSIN(PHI)*
     & DSIN(THETA)*P(2, I) + DCOS(THETA)*P(3, I) !p_k
          IF (ABS(PROT(3, I)/PROT(0, I)) < EPSILON) THEN
            PROT(3, I) = 0D0
          ENDIF
      ENDDO
      
      END