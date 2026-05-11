################################################################################
#
# Copyright (c) 2009 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################

"""Unit test library for the spin correlated decay routines
in the madspin directory"""

from __future__ import absolute_import
import sys
import os
import string
import shutil
pjoin = os.path.join

from subprocess import Popen, PIPE, STDOUT

root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
sys.path.insert(0, os.path.join(root_path,'..','..'))

import tests.unit_tests as unittest
import madgraph.interface.master_interface as Cmd
import madgraph.various.banner as banner

import copy
import array

import madgraph.core.base_objects as MG
import madgraph.various.misc as misc
import MadSpin.decay as madspin 
import models.import_ufo as import_ufo


from madgraph import MG5DIR
#
class TestBanner(unittest.TestCase):
    """Test class for the reading of the banner"""

    def test_extract_info(self):
        """Test that the banner is read properly"""

        path=pjoin(MG5DIR, 'tests', 'input_files', 'tt_banner.txt')
        inputfile = open(path, 'r')
        mybanner = banner.Banner(inputfile)
#        mybanner.ReadBannerFromFile()
        process=mybanner.get("generate")
        model=mybanner.get("model")
        self.assertEqual(process,"p p > t t~ @1")
        self.assertEqual(model,"sm")
        
    
    def test_get_final_state_particle(self):
        """test that we find the final state particles correctly"""

        cmd = Cmd.MasterCmd()
        cmd.do_import('sm')
        fct = lambda x: cmd.get_final_part(x)
        
        # 
        self.assertEqual(set([11, -11]), fct('p p > e+ e-'))
        self.assertEqual(set([11, 24]), fct('p p > w+ e-'))
        self.assertEqual(set([11, 24]), fct('p p > W+ e-'))
        self.assertEqual(set([1, 2, 3, 4, -1, 11, 21, -4, -3, -2]), fct('p p > W+ e-, w+ > j j'))
        self.assertEqual(fct('p p > t t~, (t > b w+, w+ > j j) ,t~ > b~ w-'), set([1, 2, 3, 4, -1, 21, -4, -3, -2,5,-5,-24]))
        self.assertEqual(fct('e+ e- > all all, all > e+ e-'), set([-11,11]))
        self.assertEqual(fct('e+ e- > j w+, j > e+ e-'), set([-11,11,24]))

    def test_get_proc_with_decay_LO(self):

        cmd = Cmd.MasterCmd()
        cmd.do_import('sm')
        
        # Note the ; at the end of the line is important!
        #1 simple case
        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~', 't> w+b', cmd._curr_model)
        self.assertEqual(['generate p p > t t~, t> w+b  --no_warning=duplicate;'],[out])

        #2 with @0
        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~ @0', 't> w+b', cmd._curr_model)
        self.assertEqual(['generate p p > t t~ , t> w+b @0 --no_warning=duplicate;'],[out])

        #3 with @0 and --no_warning=duplicate
        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~ @0 --no_warning=duplicate', 't> w+b', cmd._curr_model)
        self.assertEqual(['generate p p > t t~ , t> w+b @0 --no_warning=duplicate;'],[out])

        #4 test with already present decay chain
        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~, t > w+ b @0 --no_warning=duplicate', 't~ > w+b', cmd._curr_model)
        self.assertEqual(['generate p p > t t~, t~ > w+b, ( t > w+ b , t~ > w+b) @0  --no_warning=duplicate;'],[out])
        
        #4 test with already present decay chain
        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~, t > w+ b, t~ > w- b~ @0 --no_warning=duplicate', 'w >  all all', cmd._curr_model)
        self.assertEqual(['generate p p > t t~, w >  all all, ( t > w+ b, w >  all all), ( t~ > w- b~ , w >  all all) @0 --no_warning=duplicate;'],[out])

        #6 case with noborn=QCD
        # This is technically not yet supported by MS, but it is nice that this functions supports it.
        out = madspin.decay_all_events.get_proc_with_decay('generate g g > h QED=1 [noborn=QCD]', 'h > b b~', cmd._curr_model)
        self.assertEqual(['add process g g > h QED=1 [sqrvirt=QCD], h > b b~  --no_warning=duplicate;'], 
                         [out]) 

        # simple case but failing initial implementation. Handle it now but raising a critical message [mute here]
        with misc.MuteLogger(['decay'], [60]):
            out = madspin.decay_all_events.get_proc_with_decay('p p > t t~', 't~ > w- b~  QCD=99, t > w+ b  QCD=99', cmd._curr_model)
            self.assertEqual(['add process p p > t t~, t~ > w- b~  QCD=99, t > w+ b  QCD=99  --no_warning=duplicate;'],[out])
        
        self.assertRaises(Exception, madspin.decay_all_events.get_proc_with_decay, 'generate p p > t t~, (t> w+ b, w+ > e+ ve)')

    def test_get_proc_with_decay_NLO(self):

        cmd = Cmd.MasterCmd()
        cmd.do_import('sm')
        
        #1 simple case
        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~ [QCD]', 't> w+b', cmd._curr_model)
        self.assertEqual(['add process p p > t t~, t> w+b  --no_warning=duplicate',
                          'define pert_QCD = -4 -3 -2 -1 1 2 3 4 21',
                          'add process p p > t t~ pert_QCD, t> w+b  --no_warning=duplicate'],
                         out.split(';')[:-1])

        #2 simple case with QED=1
        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~ QED=1 [QCD]', 't> w+b', cmd._curr_model)
        self.assertEqual(['add process p p > t t~ QED=1, t> w+b  --no_warning=duplicate',
                          'define pert_QCD = -4 -3 -2 -1 1 2 3 4 21',
                          'add process p p > t t~ pert_QCD QED=1, t> w+b  --no_warning=duplicate'],
                         out.split(';')[:-1])

        #3 simple case with options
        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~ QED=1 [QCD] --test', 't> w+b', cmd._curr_model)
        self.assertEqual(['add process p p > t t~ QED=1, t> w+b  --no_warning=duplicate --test',
                          'define pert_QCD = -4 -3 -2 -1 1 2 3 4 21',
                          'add process p p > t t~ pert_QCD QED=1, t> w+b  --no_warning=duplicate --test'],
                         out.split(';')[:-1])

        #4 case with LOonly
        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~ QED=1 [LOonly]', 't> w+b', cmd._curr_model)
        self.assertEqual(['add process p p > t t~ QED=1, t> w+b  --no_warning=duplicate'],
                         out.split(';')[:-1])

        #5 case with LOonly=QCD
        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~ QED=1 [LOonly=QCD]', 't> w+b', cmd._curr_model)
        self.assertEqual(['add process p p > t t~ QED=1, t> w+b  --no_warning=duplicate'],
                         out.split(';')[:-1])

        #5 case with LOonly=QCD
        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~ QED=1 [LOonly=QCD,QED]', 't> w+b', cmd._curr_model)
        self.assertEqual(['add process p p > t t~ QED=1, t> w+b  --no_warning=duplicate'],
                         out.split(';')[:-1])

        #5 case with LOonly=QCD
        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~ QED=1 [LOonly=QCD QED]', 't> w+b', cmd._curr_model)
        self.assertEqual(['add process p p > t t~ QED=1, t> w+b  --no_warning=duplicate'],
                         out.split(';')[:-1])
        
        
        #6 case with all=QCD
        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~ QED=1 [all=QCD]', 't> w+b', cmd._curr_model)
        self.assertEqual(['add process p p > t t~ QED=1, t> w+b  --no_warning=duplicate',
                          'define pert_QCD = -4 -3 -2 -1 1 2 3 4 21',
                          'add process p p > t t~ pert_QCD QED=1, t> w+b  --no_warning=duplicate'],
                         out.split(';')[:-1])       

        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~ QED=1 [ all= QCD]', 't> w+b', cmd._curr_model)
         
        self.assertEqual(['add process p p > t t~ QED=1, t> w+b  --no_warning=duplicate',
                          'define pert_QCD = -4 -3 -2 -1 1 2 3 4 21',
                          'add process p p > t t~ pert_QCD QED=1, t> w+b  --no_warning=duplicate'],
                         out.split(';')[:-1])       

        #6 case with virt=QCD, technically not valid but I like that the function can do it
        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~ QED=1 [virt=QCD]', 't> w+b', cmd._curr_model)
        self.assertEqual(['add process p p > t t~ QED=1 [virt=QCD], t> w+b  --no_warning=duplicate'],
                         out.split(';')[:-1])       

          


class TestEvent(unittest.TestCase):
    """Test class for the reading of the lhe input file"""
    
    
    def test_madspin_event(self):
        """check the reading/writting of the events inside MadSpin"""
        
        inputfile = open(pjoin(MG5DIR, 'tests', 'input_files', 'madspin_event.lhe'))
        
        events = madspin.Event(inputfile)
        
        # First event
        event = events.get_next_event()
        self.assertEqual(event, 1)
        event = events
        self.assertEqual(event.string_event_compact(), """21 0.0 0 586.8395 586.84 0.7505772
21 0.0 0 -182.0876 182.0891 0.7488873
6 197.60403 48.42486 76.8186 277.8892 173
-6 -212.77359 -34.66934 359.4546 453.4437 173
21 15.169561 -13.75551 -31.52123 37.59628 0.7499895
""")
#        21 0.0 0.0 586.83954 586.84002    0.750577236977    
#21 0.0 0.0 -182.0876 182.08914    0.748887294316    
#6 197.60403 48.424858 76.818601 277.88922    173.00000459    
#-6 -212.77359 -34.669345 359.45458 453.44366    172.999981581    
#21 15.169561 -13.755513 -31.521232 37.59628    0.749989476383 
        self.assertEqual(event.get_tag(), (((21, 21), (-6, 6, 21)), [[21, 21], [6, -6, 21]]))   
        event.assign_scale_line("8 3 0.1 125 0.1 0.3")
        event.change_wgt(factor=0.4)
        
        self.assertEqual(event.string_event().split('\n'), """<event>
  8      3 +4.0000000e-02 1.25000000e+02 1.00000000e-01 3.00000000e-01
       21 -1    0    0  503  502 +0.00000000000e+00 +0.00000000000e+00 +5.86839540000e+02  5.86840020000e+02  7.50000000000e-01 0.0000e+00 0.0000e+00
       21 -1    0    0  501  503 +0.00000000000e+00 +0.00000000000e+00 -1.82087600000e+02  1.82089140000e+02  7.50000000000e-01 0.0000e+00 0.0000e+00
        6  1    1    2  504    0 +1.97604030000e+02 +4.84248580000e+01 +7.68186010000e+01  2.77889220000e+02  1.73000000000e+02 0.0000e+00 0.0000e+00
       -6  1    1    2    0  502 -2.12773590000e+02 -3.46693450000e+01 +3.59454580000e+02  4.53443660000e+02  1.73000000000e+02 0.0000e+00 0.0000e+00
       21  1    1    2  501  504 +1.51695610000e+01 -1.37555130000e+01 -3.15212320000e+01  3.75962800000e+01  7.50000000000e-01 0.0000e+00 0.0000e+00
#aMCatNLO 2  5  3  3  1 0.45933500E+02 0.45933500E+02 9  0  0 0.99999999E+00 0.69338413E+00 0.14872513E+01 0.00000000E+00 0.00000000E+00
  <rwgt>
   <wgt id='1001'>  +1.2946800e+02 </wgt>
   <wgt id='1002'>  +1.1581600e+02 </wgt>
   <wgt id='1003'>  +1.4560400e+02 </wgt>
   <wgt id='1004'>  +1.0034800e+02 </wgt>
   <wgt id='1005'>  +8.9768000e+01 </wgt>
   <wgt id='1006'>  +1.1285600e+02 </wgt>
   <wgt id='1007'>  +1.7120800e+02 </wgt>
   <wgt id='1008'>  +1.5316000e+02 </wgt>
   <wgt id='1009'>  +1.9254800e+02 </wgt>
</rwgt>
</event> 
""".split('\n'))
        
        # Second event
        event = events.get_next_event()    
        self.assertEqual(event, 1)
        event =events
        self.assertEqual(event.get_tag(), (((21, 21), (-6, 6, 21)), [[21, 21], [6, 21, -6]]))
                
        self.assertEqual(event.string_event().split('\n'), """<event>
  5     66 +3.2366351e+02 4.39615290e+02 7.54677160e-03 1.02860750e-01
       21 -1    0    0  503  502 +0.00000000000e+00 +0.00000000000e+00 +1.20582240000e+03  1.20582260000e+03  7.50000000000e-01 0.0000e+00 0.0000e+00
       21 -1    0    0  501  503 +0.00000000000e+00 +0.00000000000e+00 -5.46836110000e+01  5.46887540000e+01  7.50000000000e-01 0.0000e+00 0.0000e+00
        6  1    1    2  501    0 -4.03786550000e+01 -1.41924320000e+02 +3.66089980000e+02  4.30956860000e+02  1.73000000000e+02 0.0000e+00 0.0000e+00
       21  1    1    2  504  502 -2.46716450000e+01 +3.98371210000e+01 +2.49924260000e+02  2.54280130000e+02  7.50000000000e-01 0.0000e+00 0.0000e+00
       -6  1    1    2    0  504 +6.50503000000e+01 +1.02087200000e+02 +5.35124510000e+02  5.75274350000e+02  1.73000000000e+02 0.0000e+00 0.0000e+00
#aMCatNLO 2  5  4  4  4 0.40498390E+02 0.40498390E+02 9  0  0 0.99999997E+00 0.68201705E+00 0.15135239E+01 0.00000000E+00 0.00000000E+00
  <mgrwgt>
  some information
  <scale> even more infor
  </mgrwgt>
  <clustering>
  blabla
  </clustering>
  <rwgt>
   <wgt id='1001'> 0.32367e+03 </wgt>
   <wgt id='1002'> 0.28621e+03 </wgt>
   <wgt id='1003'> 0.36822e+03 </wgt>
   <wgt id='1004'> 0.24963e+03 </wgt>
   <wgt id='1005'> 0.22075e+03 </wgt>
   <wgt id='1006'> 0.28400e+03 </wgt>
   <wgt id='1007'> 0.43059e+03 </wgt>
   <wgt id='1008'> 0.38076e+03 </wgt>
   <wgt id='1009'> 0.48987e+03 </wgt>
  </rwgt>
</event> 
""".split('\n'))
        
        # Third event ! Not existing
        event = events.get_next_event()
        self.assertEqual(event, "no_event")
        



#class Testtopo(unittest.TestCase):
#    """Test the extraction of the topologies for the undecayed process"""
#
#    def test_topottx(self):
#
#        os.environ['GFORTRAN_UNBUFFERED_ALL']='y'
#        path_for_me=pjoin(MG5DIR, 'tests','unit_tests','madspin')
#        shutil.copyfile(pjoin(MG5DIR, 'tests','input_files','param_card_sm.dat'),\
#		pjoin(path_for_me,'param_card.dat'))
#        curr_dir=os.getcwd()
#        os.chdir('/tmp')
#        temp_dir=os.getcwd()
#        mgcmd=Cmd.MasterCmd()
#        process_prod=" g g > t t~ "
#        process_full=process_prod+", ( t > b w+ , w+ > mu+ vm ), "
#        process_full+="( t~ > b~ w- , w- > mu- vm~ ) "
#        decay_tools=madspin.decay_misc()
#        topo=decay_tools.generate_fortran_me([process_prod],"sm",0, mgcmd, path_for_me)
#        decay_tools.generate_fortran_me([process_full],"sm", 1,mgcmd, path_for_me)
#
#        prod_name=decay_tools.compile_fortran_me_production(path_for_me)
#	decay_name = decay_tools.compile_fortran_me_full(path_for_me)
#
#
#        topo_test={1: {'branchings': [{'index_propa': -1, 'type': 's',\
#                'index_d2': 3, 'index_d1': 4}], 'get_id': {}, 'get_momentum': {}, \
#                'get_mass2': {}}, 2: {'branchings': [{'index_propa': -1, 'type': 't', \
#                'index_d2': 3, 'index_d1': 1}, {'index_propa': -2, 'type': 't', 'index_d2': 4,\
#                 'index_d1': -1}], 'get_id': {}, 'get_momentum': {}, 'get_mass2': {}}, \
#                   3: {'branchings': [{'index_propa': -1, 'type': 't', 'index_d2': 4, \
#                'index_d1': 1}, {'index_propa': -2, 'type': 't', 'index_d2': 3, 'index_d1': -1}],\
#                 'get_id': {}, 'get_momentum': {}, 'get_mass2': {}}}
#        
#        self.assertEqual(topo,topo_test)
#  
#
#        p_string='0.5000000E+03  0.0000000E+00  0.0000000E+00  0.5000000E+03  \n'
#        p_string+='0.5000000E+03  0.0000000E+00  0.0000000E+00 -0.5000000E+03 \n'
#        p_string+='0.5000000E+03  0.1040730E+03  0.4173556E+03 -0.1872274E+03 \n'
#        p_string+='0.5000000E+03 -0.1040730E+03 -0.4173556E+03  0.1872274E+03 \n'        
#
#       
#        os.chdir(pjoin(path_for_me,'production_me','SubProcesses',prod_name))
#        executable_prod="./check"
#        external = Popen(executable_prod, stdout=PIPE, stdin=PIPE, stderr=STDOUT)
# 
#        external.stdin.write(p_string)
#
#        info = int(external.stdout.readline())
#        nb_output = abs(info)+1
#
#
#        prod_values = ' '.join([external.stdout.readline() for i in range(nb_output)])
#
#        prod_values=prod_values.split()
#        prod_values_test=['0.59366146660637686', '7.5713552297679376', '12.386583104018380', '34.882849897228873']
#        self.assertEqual(prod_values,prod_values_test)               
#        external.terminate()
#
#
#        os.chdir(temp_dir)
#        
#        p_string='0.5000000E+03  0.0000000E+00  0.0000000E+00  0.5000000E+03 \n'
#        p_string+='0.5000000E+03  0.0000000E+00  0.0000000E+00 -0.5000000E+03 \n'
#        p_string+='0.8564677E+02 -0.8220633E+01  0.3615807E+02 -0.7706033E+02 \n'
#        p_string+='0.1814001E+03 -0.5785084E+02 -0.1718366E+03 -0.5610972E+01 \n'
#        p_string+='0.8283621E+02 -0.6589913E+02 -0.4988733E+02  0.5513262E+01 \n'
#        p_string+='0.3814391E+03  0.1901552E+03  0.2919968E+03 -0.1550888E+03 \n'
#        p_string+='0.5422284E+02 -0.3112810E+02 -0.7926714E+01  0.4368438E+02\n'
#        p_string+='0.2144550E+03 -0.2705652E+02 -0.9850424E+02  0.1885624E+03\n'
#
#        os.chdir(pjoin(path_for_me,'full_me','SubProcesses',decay_name))
#        executable_decay="./check"
#        external = Popen(executable_decay, stdout=PIPE, stdin=PIPE, stderr=STDOUT)
#        external.stdin.write(p_string)
#
#        nb_output =1 
#        decay_value = ' '.join([external.stdout.readline() for i in range(nb_output)])
#
#        decay_value=decay_value.split()
#        decay_value_test=['3.8420345719455465E-017']
#        for i in range(len(decay_value)): 
#            self.assertAlmostEqual(eval(decay_value[i]),eval(decay_value_test[i]))
#        os.chdir(curr_dir)
#        external.terminate()
#        shutil.rmtree(pjoin(path_for_me,'production_me'))
#        shutil.rmtree(pjoin(path_for_me,'full_me'))
#        os.remove(pjoin(path_for_me,'param_card.dat'))
#        os.environ['GFORTRAN_UNBUFFERED_ALL']='n'

        
class TestGetCompatibleFlavorData(unittest.TestCase):
    """Tests for decay_all_events.get_compatible_flavor_data.

    The method should return (compatible_indices, rel_brs) where
    compatible_indices lists every flavor_index whose tuple's
    production-particle positions match the current event, and
    rel_brs carries one BR factor per compatible index.

    For p p > w+ w- (w+ > j j, w- > j j) a production event with
    u d~ initial state should yield 4 compatible entries when each
    of the four decay-product combinations (ud~/ud~, ud~/cs~,
    cs~/ud~, cs~/cs~) occupies its own flavor_groups_full entry.
    """

    # ---------- helpers ---------------------------------------------------- #

    def _make_event(self, pid_list):
        """Return a minimal Event whose .particle dict contains the given PIDs.

        pid_list: sequence of PDG codes in event order (1-indexed internally).
        """
        ev = madspin.Event()
        for i, pid in enumerate(pid_list, 1):
            ev.particle[i] = {'pid': pid}
        return ev

    def _make_handler(self, curr_event):
        """Return a bare decay_all_events instance with only curr_event set."""
        obj = madspin.decay_all_events.__new__(madspin.decay_all_events)
        object.__setattr__(obj, 'curr_event', curr_event)
        object.__setattr__(obj, 'all_ME', {})
        return obj

    # ---------- individual tests ------------------------------------------- #

    def test_no_flavor_groups_returns_single_entry(self):
        """Without flavor_groups_full the function falls back to a length-1 result."""
        ev = self._make_event([2, -1])      # u, d~
        handler = self._make_handler(ev)

        # decay_me with no flavor group data
        decay_me = {'br': 0.15, 'prod2full': [1, 2]}
        event_map = {0: 0, 1: 1}

        indices, brs = handler.get_compatible_flavor_data(
            None, decay_me, event_map)

        # Must return exactly one entry (the fallback)
        self.assertEqual(len(indices), 1)
        self.assertEqual(len(brs), 1)
        # BR must be preserved unchanged
        self.assertAlmostEqual(brs[0], 0.15)

    def test_single_matching_group(self):
        """With one matching group the function returns exactly that group."""
        ev = self._make_event([2, -1])      # u, d~ in the event
        handler = self._make_handler(ev)

        # Full ME for u d~ > u d~ u~ d  (positions 0,1 = initial state)
        # prod2full: initial u->1, initial d~->2, W+ decays, W- decays
        prod2full = [1, 2, -2, -4]
        flavor_groups_full = [
            [(2, -1, 2, -1, -2, 1)],   # u d~ initial, W+->ud~, W-->u~d
        ]
        decay_me = {'br': 0.10, 'prod2full': prod2full,
                    'flavor_groups_full': flavor_groups_full}
        event_map = {0: 0, 1: 1}

        indices, brs = handler.get_compatible_flavor_data(
            None, decay_me, event_map)

        self.assertEqual(indices, [1])
        self.assertEqual(len(brs), 1)
        self.assertAlmostEqual(brs[0], 0.10)

    def test_pp_ww_jj_jj_four_compatible_groups(self):
        """Core regression test for p p > w+ w- (w+ > j j, w- > j j).

        With u/d initial state there are four decay-product combinations:
          group 1: W+ -> u d~,  W- -> u~ d
          group 2: W+ -> u d~,  W- -> c~ s
          group 3: W+ -> c s~,  W- -> u~ d
          group 4: W+ -> c s~,  W- -> c~ s

        All four share the same production-level initial state (u, d~),
        so all four must appear in the returned indices.

        A d u~ event is incompatible with any of these groups and must
        NOT appear.
        """
        # Production event: u (pid=2) in slot 0, d~ (pid=-1) in slot 1
        ev_ud = self._make_event([2, -1])
        handler = self._make_handler(ev_ud)

        # prod2full: production particles 0,1 are initial-state (at full-ME
        # positions 1,2); particles 2,3 (W+,W-) decay (negative entries).
        prod2full = [1, 2, -2, -4]

        # Four separate flavor groups – one per decay-product combination.
        # Tuple layout: (init1, init2, decay1_d1, decay1_d2, decay2_d1, decay2_d2)
        # positions 0,1 = initial state  (checked against pos_to_pid)
        # positions 2,3,4,5 = decay products (not constrained by production event)
        flavor_groups_full = [
            [(2, -1,  2, -1, -2,  1)],   # W+ -> u d~,  W- -> u~ d
            [(2, -1,  2, -1, -4,  3)],   # W+ -> u d~,  W- -> c~ s
            [(2, -1,  4, -3, -2,  1)],   # W+ -> c s~,  W- -> u~ d
            [(2, -1,  4, -3, -4,  3)],   # W+ -> c s~,  W- -> c~ s
            [(1, -2,  1, -2, -1,  2)],   # d u~ initial (incompatible)
        ]

        decay_me = {'br': 0.12, 'prod2full': prod2full,
                    'flavor_groups_full': flavor_groups_full}
        event_map = {0: 0, 1: 1}

        indices, brs = handler.get_compatible_flavor_data(
            None, decay_me, event_map)

        # Must find exactly 4 compatible groups (groups 1-4; group 5 is d u~)
        self.assertEqual(sorted(indices), [1, 2, 3, 4],
            msg="Expected 4 compatible flavor groups for u d~ event, got %s" % indices)
        self.assertEqual(len(brs), 4)
        for br in brs:
            self.assertAlmostEqual(br, 0.12)

    def test_pp_ww_incompatible_initial_state_excluded(self):
        """Groups whose initial-state PDGs mismatch the event are excluded."""
        # Production event: d (pid=1) in slot 0, u~ (pid=-2) in slot 1
        ev_du = self._make_event([1, -2])
        handler = self._make_handler(ev_du)

        prod2full = [1, 2, -2, -4]
        flavor_groups_full = [
            [(2, -1,  2, -1, -2,  1)],   # u d~ initial (incompatible with d u~)
            [(2, -1,  2, -1, -4,  3)],
            [(2, -1,  4, -3, -2,  1)],
            [(2, -1,  4, -3, -4,  3)],
            [(1, -2,  1, -2, -1,  2)],   # d u~ initial (compatible)
        ]
        decay_me = {'br': 0.12, 'prod2full': prod2full,
                    'flavor_groups_full': flavor_groups_full}
        event_map = {0: 0, 1: 1}

        indices, brs = handler.get_compatible_flavor_data(
            None, decay_me, event_map)

        # Only group 5 (d u~ initial state) is compatible
        self.assertEqual(indices, [5])
        self.assertEqual(len(brs), 1)

    def test_old_grouped_structure_returns_length_one(self):
        """Document pre-fix behaviour: when decay combinations are lumped into
        a single flavor group (as get_external_flavors_with_iden used to do),
        get_compatible_flavor_data returns only 1 entry for a u d~ event.

        This test captures the former behaviour so the regression is visible.
        After the fix (using individual tuples as separate groups) the test
        test_pp_ww_jj_jj_four_compatible_groups replaces this case.
        """
        ev_ud = self._make_event([2, -1])
        handler = self._make_handler(ev_ud)

        prod2full = [1, 2, -2, -4]
        # Old behaviour: all four decay combinations in ONE group
        flavor_groups_full = [
            [
                (2, -1,  2, -1, -2,  1),
                (2, -1,  2, -1, -4,  3),
                (2, -1,  4, -3, -2,  1),
                (2, -1,  4, -3, -4,  3),
            ],  # group 1 – all u d~ initial variants lumped together
            [
                (1, -2,  1, -2, -1,  2),
            ],  # group 2 – d u~ initial state
        ]
        decay_me = {'br': 0.12, 'prod2full': prod2full,
                    'flavor_groups_full': flavor_groups_full}
        event_map = {0: 0, 1: 1}

        indices, brs = handler.get_compatible_flavor_data(
            None, decay_me, event_map)

        # With the old grouped structure only 1 group is returned –
        # this is the bug: we cannot distinguish the 4 decay combinations.
        self.assertEqual(len(indices), 1,
            msg="Old grouped structure returns 1 (bug: cannot distinguish "
                "decay combinations). With per-tuple groups this would be 4.")

    def test_br_preserved_and_not_normalized(self):
        """rel_brs must equal decay_me['br'] without normalisation."""
        ev = self._make_event([2, -1])
        handler = self._make_handler(ev)

        prod2full = [1, 2, -2, -4]
        # Two compatible groups for u d~ initial state
        flavor_groups_full = [
            [(2, -1, 2, -1, -2, 1)],
            [(2, -1, 4, -3, -2, 1)],
        ]
        br_value = 0.347
        decay_me = {'br': br_value, 'prod2full': prod2full,
                    'flavor_groups_full': flavor_groups_full}
        event_map = {0: 0, 1: 1}

        _, brs = handler.get_compatible_flavor_data(
            None, decay_me, event_map)

        self.assertEqual(len(brs), 2)
        for br in brs:
            self.assertAlmostEqual(br, br_value,
                msg="BR must not be normalised; expected %s, got %s"
                    % (br_value, br))

    def test_get_flavor_data_from_me_produces_per_tuple_groups(self):
        """get_flavor_data_from_me must assign one flavor_index per flavor tuple.

        This is the key requirement for get_compatible_flavor_data to return
        length 4 for p p > w+ w- (w+ > j j, w- > j j):
        each decay-product combination must be a separate Fortran entry.

        We mock the matrix element with four flavor tuples that all share the
        same coupling structure (as get_external_flavors_with_iden would group
        them together) but must be split into four separate flavor_groups entries.
        """
        # Minimal mock of a HelasMatrixElement
        class MockModel:
            def get(self, key):
                return {}  # no merged particles

        class MockProcess:
            def get(self, key):
                if key == 'model':
                    return MockModel()
                return None

        # Four distinct flavor tuples, all with the same coupling in the old scheme
        # (representing the four w+ > j j / w- > j j combinations for u d~ initial)
        flavor_tuples = [
            (2, -1,  2, -1, -2,  1),   # W+ -> u d~,  W- -> u~ d
            (2, -1,  2, -1, -4,  3),   # W+ -> u d~,  W- -> c~ s
            (2, -1,  4, -3, -2,  1),   # W+ -> c s~,  W- -> u~ d
            (2, -1,  4, -3, -4,  3),   # W+ -> c s~,  W- -> c~ s
        ]

        # Single coupling key – simulates the old behaviour where all four
        # tuples would be placed in the same group
        common_coupling = ('coupling_A',)

        class MockME:
            def get_nexternal_ninitial(self):
                return (6, 2)

            def get(self, key):
                if key == 'processes':
                    return [MockProcess()]
                return None

            def get_external_flavors(self):
                return list(flavor_tuples)

            def get_external_flavors_with_iden(self):
                # All four tuples share the same coupling – old code would put
                # them all in one group
                return {common_coupling: list(flavor_tuples)}.values()

        nexternal, flavor_combos, _, flavor_groups = \
            madspin.decay_all_events.get_flavor_data_from_me(MockME())

        # Each group must contain exactly one tuple (no coupling-based merging)
        self.assertEqual(len(flavor_groups), len(flavor_tuples),
            msg="Expected one group per flavor tuple (%d tuples), got %d groups"
                % (len(flavor_tuples), len(flavor_groups)))
        for i, grp in enumerate(flavor_groups):
            self.assertEqual(len(grp), 1,
                msg="flavor_groups[%d] has %d tuples; expected exactly 1" % (i, len(grp)))

        # flavor_combos must also be one entry per tuple
        self.assertEqual(len(flavor_combos), len(flavor_tuples))

