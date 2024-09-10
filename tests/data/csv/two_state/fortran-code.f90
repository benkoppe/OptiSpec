      program dsdrv1 
!

!     Authors for the part of Diagonalization of Hamiltonian
!     Richard Lehoucq
!     Danny Sorensen
!     Chao Yang
!     Dept. of Computational &
!     Applied Mathematics
!     Rice University
!     Houston, Texas
!
!     The part related to spectra and Hamiltonian is written by Xian-Kai Chen 
!     (chenxiankai2009@gmail.com)
!--------------------------------------------------------------------------------
!
!     %-----------------------------%
!     | Define leading dimensions   |
!     | for all arrays.             |
!     | MAXN:   Maximum dimension   |
!     |         of the A allowed.   |
!     | MAXNEV: Maximum NEV allowed |
!     | MAXNCV: Maximum NCV allowed |
!     %-----------------------------%
!
      integer              maxn, maxnev, maxncv, ldv
      parameter            (maxn=2*20*200, maxnev=4000, maxncv=5000, &
     &                     ldv=maxn )
      integer,parameter :: nn=20*200
!     %--------------%
!     | Local Arrays |
!     %--------------%
!
      Double precision  &
     &                 v(ldv,maxncv), workl(maxncv*(maxncv+8)),&
     &                 workd(3*maxn), d(maxncv,2), resid(maxn),&
     &                 ax(maxn)
      logical          select(maxncv)
      integer          iparam(11), ipntr(11)
!
!     %---------------%
!     | Local Scalars |
!     %---------------%
!
      character        bmat*1, which*2
      integer          ido, n, nev, ncv, lworkl, info, ierr, j,& 
     &                 nx, nconv, maxitr, mode, ishfts
      logical          rvec
      Double precision   &   
     &                 tol, sigma
      integer           jj, je1, je2
      integer           k1, k2, kk2, kkk2 
      character(3)      c  
      real(8)           e2, ee2, optical 
      integer,parameter :: maxe2=2001
      real(8),parameter :: gama=0.0d0,sita=0.0d0                                                            
      
      real(8)           A_e2(maxe2), A_optical(maxe2)
      real(8)           r2     
      
      real(8)           Ttdm, Ttdm_X
      real(8)           tdm22                                                                                    
      real(8)           dm22_X
     
      
      real(8)           tempT
      real(8)           kbT                                                                                                     

!
!     %------------%
!     | Parameters |
!     %------------%
!
      Double precision  &
     &                 zero
      parameter        (zero = 0.0D+0)
!  
!     %-----------------------------%
!     | BLAS & LAPACK routines used |
!     %-----------------------------%
!
      Double precision  &         
     &                 dnrm2
      external         dnrm2, daxpy
!
!     %--------------------%
!     | Intrinsic function |
!     %--------------------%
!
      intrinsic        abs
!
!     %-----------------------%
!     | Executable Statements |
!     %-----------------------%
!
!     %----------------------------------------------------%
!     | The number NX is the number of interior points     |
!     | in the discretization of the 2-dimensional         |
!     | Laplacian on the unit square with zero Dirichlet   |
!     | boundary condition.  The number N(=NX*NX) is the   |
!     | dimension of the matrix.  A standard eigenvalue    |
!     | problem is solved (BMAT = 'I'). NEV is the number  |
!     | of eigenvalues to be approximated.  The user can   |
!     | modify NEV, NCV, WHICH to solve problems of        |
!     | different sizes, and to get different parts of the |
!     | spectrum.  However, The following conditions must  |
!     | be satisfied:                                      |
!     |                   N <= MAXN,                       | 
!     |                 NEV <= MAXNEV,                     |
!     |             NEV + 1 <= NCV <= MAXNCV               | 
!     %----------------------------------------------------% 
!
      n = 2*20*200
      nev =  4000
      ncv =  5000
      if ( n .gt. maxn ) then
         print *, ' ERROR with _SDRV1: N is greater than MAXN '
         go to 9000
      else if ( nev .gt. maxnev ) then
         print *, ' ERROR with _SDRV1: NEV is greater than MAXNEV '
         go to 9000
      else if ( ncv .gt. maxncv ) then
         print *, ' ERROR with _SDRV1: NCV is greater than MAXNCV '
         go to 9000
      end if
      bmat = 'I'
      which = 'SA'
!
!     %--------------------------------------------------%
!     | The work array WORKL is used in DSAUPD as        |
!     | workspace.  Its dimension LWORKL is set as       |
!     | illustrated below.  The parameter TOL determines |
!     | the stopping criterion.  If TOL<=0, machine      |
!     | precision is used.  The variable IDO is used for |
!     | reverse communication and is initially set to 0. |
!     | Setting INFO=0 indicates that a random vector is |
!     | generated in DSAUPD to start the Arnoldi         |
!     | iteration.                                       |
!     %--------------------------------------------------%
!
      lworkl = ncv*(ncv+8)
      tol = zero 
      info = 0
      ido = 0
!
!     %---------------------------------------------------%
!     | This program uses exact shifts with respect to    |
!     | the current Hessenberg matrix (IPARAM(1) = 1).    |
!     | IPARAM(3) specifies the maximum number of Arnoldi |
!     | iterations allowed.  Mode 1 of DSAUPD is used     |
!     | (IPARAM(7) = 1).  All these options may be        |
!     | changed by the user. For details, see the         |
!     | documentation in DSAUPD.                          |
!     %---------------------------------------------------%
!
      ishfts = 1
      maxitr = 300
      mode   = 1
!      
      iparam(1) = ishfts 
      iparam(3) = maxitr 
      iparam(7) = mode 
!
!     %-------------------------------------------%
!     | M A I N   L O O P (Reverse communication) |
!     %-------------------------------------------%

      jj=1                                              ! job number                                                                             
      write(c,'(i3)') jj
      open(12,file='2_states_energy_'//trim(adjustl(c))//'.dat',status='unknown')

 10   continue
!
!        %---------------------------------------------%
!        | Repeatedly call the routine DSAUPD and take | 
!        | actions indicated by parameter IDO until    |
!        | either convergence is indicated or maxitr   |
!        | has been exceeded.                          |
!        %---------------------------------------------%
!
         call dsaupd ( ido, bmat, n, which, nev, tol, resid, &
     &                 ncv, v, ldv, iparam, ipntr, workd, workl,  &
     &                 lworkl, info )
!
         if (ido .eq. -1 .or. ido .eq. 1) then
!
!           %--------------------------------------%
!           | Perform matrix vector multiplication |
!           |              y <--- OP*x             |
!           | The user should supply his/her own   |
!           | matrix vector multiplication routine |
!           | here that takes workd(ipntr(1)) as   |
!           | the input, and return the result to  |
!           | workd(ipntr(2)).                     |
!           %--------------------------------------%
!
            call av (n,  workd(ipntr(1)), workd(ipntr(2)))
!
!           %-----------------------------------------%
!           | L O O P   B A C K to call DSAUPD again. |
!           %-----------------------------------------%
!
            go to 10
!
         end if 
!
!     %----------------------------------------%
!     | Either we have convergence or there is |
!     | an error.                              |
!     %----------------------------------------%
!
      if ( info .lt. 0 ) then
!
!        %--------------------------%
!        | Error message. Check the |
!        | documentation in DSAUPD. |
!        %--------------------------%
!
         print *, ' '
         print *, ' Error with _saupd, info = ', info
         print *, ' Check documentation in _saupd '
         print *, ' '
!
      else 
!
!        %-------------------------------------------%
!        | No fatal errors occurred.                 |
!        | Post-Process using DSEUPD.                |
!        |                                           |
!        | Computed eigenvalues may be extracted.    |  
!        |                                           |
!        | Eigenvectors may also be computed now if  |
!        | desired.  (indicated by rvec = .true.)    | 
!        %-------------------------------------------%
!           
         rvec = .true.
!
         call dseupd ( rvec, 'All', select, d, v, ldv, sigma, &
     &        bmat, n, which, nev, tol, resid, ncv, v, ldv, &
     &        iparam, ipntr, workd, workl, lworkl, ierr )
     
      write(12,15) d(1:nev,1)-d(1,1)
  15  format(f18.9) 
         
      open(14,file='2_states_absorption_'//trim(adjustl(c))//'.dat',status='unknown')                      
      

      tempT = 300.0d0                                                       !temp in K, to convert to thermal E
      kbT = tempT * 0.695028d0                                            !thermal energy
      
      tdm22 = 38.0d0                                                      !dipole moment of diabatic CT state

      r2 = 200.0d0                                                       ! original 100
      ee2 = 10.0d0                                                        !vibronic peak expansion width                                                                   

      A_e2=0.0d0
      A_optical=0.0d0

         e2 = -ee2
         do je2=1,maxe2

           e2 = e2 + ee2
           optical = 0.0d0

            do kk2=1,nev                                                  !final   state beta
              do k2=1,50                                                  !initial state alpha
                    
                    dm22_X = 0.0d0 
               
                    Ttdm   = 0.0d0
!---------------------------------------------------------------------------------------------------------------------------------             
               if((d(kk2,1)-d(k2,1)).gt.0.0d0)    then
   
                 do kkk2=1,nn                                           
                    dm22_X = dm22_X + tdm22  * v(nn+kkk2,k2)*v(nn+kkk2,kk2)
                                            
                 enddo
                   
                 Ttdm   = (dm22_X)**2
             
             !   optical = optical + (Ttdm) * exp(-(e2-(d(kk2,1)-d(k2,1)))**2/r2**2)             ! spectra at 0 K                                                                          
                 optical = optical + (exp(-d(k2,1)/kbT) - exp(-d(kk2,1)/kbT)) * (Ttdm) * exp(-(e2-(d(kk2,1)-d(k2,1)))**2/r2**2)
               endif
!--------------------------------------------------------------------------------------------------------------------------------------
              enddo
            enddo
        
                 A_e2(je2)      = e2
                 A_optical(je2) = optical
           enddo

! Abs-normalization
      !     A_optical = A_optical / maxval(A_optical)

          write(14,17) (A_optical(je2), je2=1,maxe2)
  17      format(1pe24.17)



!        %----------------------------------------------%
!        | Eigenvalues are returned in the first column |
!        | of the two dimensional array D and the       |
!        | corresponding eigenvectors are returned in   |
!        | the first NEV columns of the two dimensional |
!        | array V if requested.  Otherwise, an         |
!        | orthogonal basis for the invariant subspace  |
!        | corresponding to the eigenvalues in D is     |
!        | returned in V.                               |
!        %----------------------------------------------%
!
         if ( ierr .ne. 0) then
!
!            %------------------------------------%
!            | Error condition:                   |
!            | Check the documentation of DSEUPD. |
!            %------------------------------------%
!
             print *, ' '
             print *, ' Error with _seupd, info = ', ierr
             print *, ' Check the documentation of _seupd. '
             print *, ' '
!
         else
!
             nconv =  iparam(5)
             do 20 j=1, nconv
!
!               %---------------------------%
!               | Compute the residual norm |
!               |                           |
!               |   ||  A*x - lambda*x ||   |
!               |                           |
!               | for the NCONV accurately  |
!               | computed eigenvalues and  |
!               | eigenvectors.  (iparam(5) |
!               | indicates how many are    |
!               | accurate to the requested |
!               | tolerance)                |
!               %---------------------------%
!
                call av(n, v(1,j), ax)
                call daxpy(n, -d(j,1), v(1,j), 1, ax, 1)
                d(j,2) = dnrm2(n, ax, 1)
                d(j,2) = d(j,2) / abs(d(j,1))
!
 20          continue
!
!            %-------------------------------%
!            | Display computed residuals    |
!            %-------------------------------%
!
             call dmout(6, nconv, 2, d, maxncv, -6,  &
     &            'Ritz values and relative residuals')
         end if
!
!        %------------------------------------------%
!        | Print additional convergence information |
!        %------------------------------------------%
!
         if ( info .eq. 1) then
            print *, ' '
            print *, ' Maximum number of iterations reached.'
            print *, ' '
         else if ( info .eq. 3) then
            print *, ' ' 
            print *, ' No shifts could be applied during implicit  &
     &                 Arnoldi update, try increasing NCV.'
            print *, ' '
         end if      
!
         print *, ' '
         print *, ' _SDRV1 '
         print *, ' ====== '
         print *, ' '
         print *, ' Size of the matrix is ', n
         print *, ' The number of Ritz values requested is ', nev
         print *, ' The number of Arnoldi vectors generated', &
     &            ' (NCV) is ', ncv
         print *, ' What portion of the spectrum: ', which
         print *, ' The number of converged Ritz values is ', &
     &              nconv 
         print *, ' The number of Implicit Arnoldi update',   &
     &            ' iterations taken is ', iparam(3)
         print *, ' The number of OP*x is ', iparam(9)
         print *, ' The convergence criterion is ', tol
         print *, ' '
!
      end if
!
!     %---------------------------%
!     | Done with program dsdrv1. |
!     %---------------------------%
!
 9000 continue
!
      end
!-------------------------------------------------------------------
SUBROUTINE av(n,v,w)
    implicit none
    integer n
    integer,parameter   :: maxm=19, maxk=199
    double precision v(n), w(n)
    call multv(maxm,maxk,v,w)
end SUBROUTINE av

SUBROUTINE multv(maxm,maxk,v,w)
    IMPLICIT NONE
    INTEGER, parameter              ::  maxn=2
    INTEGER, intent(in)             ::  maxm, maxk
    DOUBLE PRECISION, intent(in)    ::  v(0:maxm,0:maxk,maxn)
    DOUBLE PRECISION, intent(out)   ::  w(0:maxm,0:maxk,maxn)
    ! Defining constants of vibrational Hamiltonian
    REAL(8)                             t0                                            ! transfer integral             
    REAL(8)                             g1, g2                                        ! e-vib coupling constants         
    REAL(8)                             f1, f2                                        ! freq              
    REAL(8)                             AE2, AE22                                     ! AE2: diabatic CT energy
    REAL(8)                             F, tdm222, aa                                     ! field energy (DELETE TDM222 LATER)


    ! quantum numbers for rows of H
    INTEGER m, k 
    ! intermediate quantities used to evaluate elements of H that depend only on m
    DOUBLE PRECISION    gm1,   &   
                        gm         
    ! intermediate quantities used to evaluate elements of H that depend only on k
    DOUBLE PRECISION    gk1,   &   
                        gk         

    F = 0.0d0                   ! / V/Angstrom (Range=0.001-0.05) (0.01 v/A)
    AE22 = 8000.0d0                ! / cm-1   
    tdm222 = 0.0d0                  ! / debye
    aa = 0.0d0                      ! / Angstrom^3 (polarizability) (Range=1-1000)
    AE2 = AE22 - (tdm222 * F * 1679.0870295) - (0.5 * F * F * aa * 559.91)            
   
    t0 = 100.0d0          ! /cm-1      
            
    g1 = 2.0d0
    g2 = 0.7d0

    f1 = 100.0d0         ! / cm-1 
    f2 = 1200.0d0          ! / cm-1            


    ! The product is decomposed into contributions from all the blocks (such as H11, H12 and H22...)
    ! (  H11    H12  ) (v1) = (w1)
    ! (  H21    H22  ) (v2) = (w2)
    !  =>  
    !    w1 = H11.v1 + H12.v2  
    !    w2 = H21.v1 + H22.v2 
    ! To calculate this, we go through each non-zero elements of all the blocks
    ! , then increment the contributions to w.

    w = 0d0

    ! Off-diagonal contributions 
      do k=0,maxk
        do m=0,maxm
        gm  = g1*f1*sqrt(dble(m))
        gm1 = g1*f1*sqrt(dble(m+1))
        gk  = g2*f2*sqrt(dble(k))
        gk1 = g2*f2*sqrt(dble(k+1))
        
                    w(m,k,1) = w(m,k,1) + v(m,k,1)*((m+0.5)*f1+(k+0.5)*f2)
        
                    w(m,k,2) = w(m,k,2) + v(m,k,2)*((m+0.5)*f1+(k+0.5)*f2+AE2+g1*g1*f1+g2*g2*f2)
        if(m>0)     w(m,k,2) = w(m,k,2) + v(m-1,k,2)*gm
        if(m<maxm)  w(m,k,2) = w(m,k,2) + v(m+1,k,2)*gm1
        if(k>0)     w(m,k,2) = w(m,k,2) + v(m,k-1,2)*gk
        if(k<maxk)  w(m,k,2) = w(m,k,2) + v(m,k+1,2)*gk1
           
                    w(m,k,1) = w(m,k,1) + v(m,k,2)*t0 
  
                    w(m,k,2) = w(m,k,2) + v(m,k,1)*t0      
      
       enddo 
     enddo 
END SUBROUTINE multv


