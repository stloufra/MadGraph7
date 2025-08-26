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
from __future__ import absolute_import
import subprocess
import unittest
import os
import re
import shutil
import sys
import logging
import tempfile

pjoin = os.path.join

logger = logging.getLogger('test_cmd')

import tests.unit_tests.iolibs.test_file_writers as test_file_writers

import madgraph.interface.master_interface as Cmd
import madgraph.interface.launch_ext_program as launch_ext
import madgraph.iolibs.files as files
import madgraph.core.diagram_generation as diagram_generation
import madgraph.various.misc as misc
import madgraph.various.lhe_parser as lhe_parser
_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
_pickle_path =os.path.join(_file_path, 'input_files')

from madgraph import MG4DIR, MG5DIR, MadGraph5Error, InvalidCmd
from tests import test_manager

#===============================================================================
# TestCmd
#===============================================================================
class TestCmdShell1(unittest.TestCase):
    """this treats all the command not related to MG_ME"""

    def setUp(self):
        """ basic building of the class to test """
        
        self.cmd = Cmd.MasterCmd()
    
    @staticmethod
    def join_path(*path):
        """join path and treat spaces"""   

        combine = os.path.join(*path)
        return combine.replace(' ',r'\ ')        
    
    def do(self, line):
        """ exec a line in the cmd under test """        
        self.cmd.exec_cmd(line)
        
    def test_generate(self):
        """command 'generate' works"""
        
        self.do('import model sm')
        self.cmd._curr_model.pass_particles_name_in_mg_default()
        self.do('generate e+ e- > e+ e- QED<=2')
        self.assertTrue(self.cmd._curr_amps)
        self.do('define P Z u')
        self.do('define J P g')
        self.do('add process e+ e- > J')
        self.assertEqual(len(self.cmd._curr_amps), 2)
        self.do('add process mu+ mu- > P, Z > mu+ mu-')
        self.assertEqual(len(self.cmd._curr_amps), 3)
        self.do('generate e+ e- > Z > e+ e-')
        self.assertEqual(len(self.cmd._curr_amps), 1)
        self.assertEqual(len(self.cmd._curr_amps[0].get('diagrams')), 1)
        # Test the "or" functionality for propagators
        self.do('define V z|a')
        self.do('generate e+ e- > V > e+ e-')
        self.assertEqual(len(self.cmd._curr_amps), 1)
        self.assertEqual(len(self.cmd._curr_amps[0].get('diagrams')), 2)
        self.do('generate e+ e- > z | a > e+ e-')
        self.assertEqual(len(self.cmd._curr_amps), 1)
        self.assertEqual(len(self.cmd._curr_amps[0].get('diagrams')), 2)
        self.do('generate d d~ > u u~ WEIGHTED^2>-1')
        self.assertEqual(len(self.cmd._curr_amps), 1)
        self.assertEqual(len(self.cmd._curr_amps[0].get('diagrams')), 4)
        self.do('generate d d~ > u u~ WEIGHTED^2>-2')
        self.assertEqual(len(self.cmd._curr_amps), 1)
        self.assertEqual(len(self.cmd._curr_amps[0].get('diagrams')), 3)
        self.assertRaises(MadGraph5Error, self.do, 
                                           'generate d d~ > u u~ WEIGHTED^2>-4')
        self.assertRaises(MadGraph5Error, self.do, 'generate a V > e+ e-')
        self.assertRaises(MadGraph5Error, self.do, 'generate e+ e+|e- > e+ e-')
        self.assertRaises(MadGraph5Error, self.do, 'generate e+ e- > V a')
        self.assertRaises(MadGraph5Error, self.do, 'generate e+ e- > e+ e- / V')
        self.do('define V2 = w+ V')
        self.assertEqual(self.cmd._multiparticles['v2'],
                         [[24, 23], [24, 22]])
        
        self.do('generate e+ ve > V2 > e+ ve mu+ mu-')
        self.assertEqual(len(self.cmd._curr_amps[0].get('diagrams')), 8)
        
        self.do('generate e+ e- > e+ e- QED=2 [tree=QCD] QCD=0')
        self.assertEqual(len(self.cmd._curr_amps[0].get('diagrams')), 4)

        self.do('generate e+ e- > e+ e- @0 QCD<=2')
        self.assertEqual(len(self.cmd._curr_amps[0].get('diagrams')), 4)   
        
        self.do('generate u u~ > d d~ QED>0')
        self.assertEqual(len(self.cmd._curr_amps[0].get('diagrams')), 3)           
        
        self.assertRaises(diagram_generation.NoDiagramException, self.do, 'generate u u~ > d d~ QED>0 QED^2==0')
        self.do('generate u u~ > d d~ QED==0 QCD>1 QED^2<=4')
        self.assertEqual(len(self.cmd._curr_amps[0].get('diagrams')), 1)
        
        self.do('generate u u~ > d d~ c c~ QED==2')
        self.assertEqual(len(self.cmd._curr_amps[0].get('diagrams')), 28)
        
            
    def test_import_model(self):
        """check that old UFO model are loaded correctly"""
        
        self.do('''import model DY_SM''')
        self.do('''import model TopEffTh''')
        self.do('''import model uutt_tch_scalar''')
        self.do('''import model uutt_sch_4fermion''')
        self.do('''import model 2HDM''')
                
    def test_draw(self):
        """ command 'draw' works """

        self.do('set group_subprocesses False')
        self.do('import model_v4 sm')
        self.do('generate e+ e- > e+ e-')
        self.do('display diagrams .')
        self.assertTrue(os.path.exists('./diagrams_0_epem_epem.eps'))
        os.remove('./diagrams_0_epem_epem.eps')
        
        self.do('generate g g > g g')
        self.do('display diagrams .')
        self.assertTrue(os.path.exists('diagrams_0_gg_gg.eps'))
        os.remove('diagrams_0_gg_gg.eps')
        self.do('set group_subprocesses True')
        
    def test_config(self):
        """check that configuration file is at default value"""
        self.maxDiff=None
        self.cmd.options = {} #reset to None
        config = self.cmd.set_configuration(MG5DIR+'/input/.mg5_configuration_default.txt', final=False)
        config =dict(config)
        del config['stdout_level']
#        for key in config.keys():
#            if key.endswith('_path') and key != 'cluster_temp_path':
#                del config[key]
        expected = {'web_browser': None, 
                    'text_editor': None, 
                    'cluster_queue': None,
                    'nb_core': None,
                    #'pjfry': 'auto',
                    'golem': 'auto',
                    'run_mode': 2,
                    'pythia-pgs_path': './pythia-pgs', 
                    'td_path': './td', 
                    'delphes_path': './Delphes', 
                    'default_unset_couplings': 99,
                    'cluster_type': 'condor', 
                    'cluster_status_update': (600, 30),
                    'madanalysis_path': './MadAnalysis', 
                    'cluster_temp_path': None, 
                    'fortran_compiler': None, 
                    'cpp_compiler': None,
                    'exrootanalysis_path': './ExRootAnalysis', 
                    'eps_viewer': None, 
                    'automatic_html_opening': True, 
                    'pythia8_path': './HEPTools/pythia8',
                    'mg5amc_py8_interface_path': './HEPTools/MG5aMC_PY8_interface',
                    'madanalysis5_path': './HEPTools/madanalysis5/madanalysis5',
                    'group_subprocesses': 'Auto',
                    'complex_mass_scheme': False,
                    'gauge': 'unitary',
                    'output_dependencies': 'external',
                    'lhapdf': 'lhapdf-config',
                    'lhapdf_py2': None,
                    'lhapdf_py3': None,  
                    'loop_optimized_output': True,
                    'fastjet': 'fastjet-config',
                    'notification_center':True,
                    'timeout': 60,
                    'ignore_six_quark_processes': False,
                    'include_lepton_initiated_processes': False,
                    'OLP': 'MadLoop',
                    'crash_on_error': False,
                    'auto_update': 7,
                    'cluster_nb_retry': 1,
                    'f2py_compiler':None,
                    'f2py_compiler_py2':None,
                    'f2py_compiler_py3':None,
                    'cluster_retry_wait': 300,
                    'syscalc_path':'./SysCalc',
                    'collier':'./HEPTools/lib',
                    'hepmc_path': './hepmc',
                    'hwpp_path': './herwigPP',
                    'thepeg_path': './thepeg',
                    #'applgrid': 'applgrid-config',
                    'pineappl': 'pineappl',
                    'cluster_size': 100,
                    'loop_color_flows': False,
                    'cluster_local_path': None,
                    'max_npoint_for_channel': 0,
                    'low_mem_multicore_nlo_generation': False,
                    'ninja': './HEPTools/lib',
                    'samurai': None,
                    'max_t_for_channel': 99,
                    'zerowidth_tchannel': True,
                    'auto_convert_model': True,
                    'nlo_mixed_expansion': True,
                    'acknowledged_v3.1_syntax': False,
                    'contur_path': './HEPTools/contur',
                    'rivet_path': './HEPTools/rivet',
                    'yoda_path':'./HEPTools/yoda',
                    'eMELA': 'eMELA-config',
                    'cluster_walltime': None,
                    }

        self.assertEqual(config, expected)
        
        #text_editor = 'vi'
        #if 'EDITOR' in os.environ and os.environ['EDITOR']:
        #    text_editor = os.environ['EDITOR']
        
        #if sys.platform == 'darwin':
        #    self.assertEqual(launch_ext.open_file.web_browser, None)
        #    self.assertEqual(launch_ext.open_file.text_editor, text_editor)
        #    self.assertEqual(launch_ext.open_file.eps_viewer, None)
        #else:
        #    self.assertEqual(launch_ext.open_file.web_browser, 'firefox')
        #    self.assertEqual(launch_ext.open_file.text_editor, text_editor)
        #    self.assertEqual(launch_ext.open_file.eps_viewer, 'gv')
                        
class TestCmdShell2(unittest.TestCase,
                    test_file_writers.CheckFileCreate):
    """Test all command line related to MG_ME"""

    debugging = True
    def setUp(self):
        
        self.cmd = Cmd.MasterCmd()
        if not self.debugging:
            self.tmpdir = tempfile.mkdtemp(prefix='amc')
        else:
            if os.path.exists(pjoin(MG5DIR, 'TEST_AMC')):
                shutil.rmtree(pjoin(MG5DIR, 'TEST_AMC'))
            os.mkdir(pjoin(MG5DIR, 'TEST_AMC'))
            self.tmpdir = pjoin(MG5DIR, 'TEST_AMC')
            
        self.out_dir = pjoin(self.tmpdir,'MGProcess')
        
        
    def tearDown(self):
        if not self.debugging and os.path.exists(self.out_dir):
            shutil.rmtree(self.out_dir)
    
    join_path = TestCmdShell1.join_path

    def do(self, line, force=False):
        """ exec a line in the cmd under te
        st """
        if force:        
            self.cmd.exec_cmd(line, force=force)
        else:   
           self.cmd.exec_cmd(line) 
    
    def test_output_madevent_directory(self):
        """Test outputting a MadEvent directory"""

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)
        
        self.cmd.do_import('model_v4 sm', force=True)
        self.do('set group_subprocesses False')
        self.do('generate e+ e- > e+ e-')
#        self.do('load processes %s' % self.join_path(_pickle_path,'e+e-_e+e-.pkl'))
        self.do('output %s -nojpeg' % self.out_dir)
        
        self.assertTrue(os.path.exists(self.out_dir))
        self.assertTrue(os.path.exists(pjoin(self.out_dir, 'Cards', 'me5_configuration.txt')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'SubProcesses', 'P0_epem_epem')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                 'Cards', 'proc_card_mg5.dat')))
        self.assertFalse(os.path.exists(os.path.join(self.out_dir,
                                                    'Cards',
                                                    'ident_card.dat')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                 'Cards', 'run_card_default.dat')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                 'Cards', 'plot_card_default.dat')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'Source',
                                                    'maxconfigs.inc')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P0_epem_epem',
                                                    'maxconfigs.inc')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P0_epem_epem',
                                                    'get_color.f')))
        if misc.which('gs'):
            self.assertFalse(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P0_epem_epem',
                                                    'matrix1.jpg')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'madevent.tar.gz')))
        self.do('output %s -f' % self.out_dir)
        self.do('set group_subprocesses True')
        if misc.which('gs'):
            self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P0_epem_epem',
                                                    'matrix1.jpg')))

        # Test the tar file
        os.mkdir(os.path.join(self.out_dir, 'temp'))
        devnull = open(os.devnull,'w')
        subprocess.call(['tar', 'xzf', os.path.join(os.path.pardir,
                                                    "madevent.tar.gz")],
                        stdout=devnull, stderr=devnull, 
                        cwd=os.path.join(self.out_dir, 'temp'))

        self.assertTrue(os.path.exists(pjoin(self.out_dir,'temp', 'Cards', 'me5_configuration.txt')))
        # Check that the Source directory compiles
        status = subprocess.call(['make'],
                                stdout=devnull, stderr=devnull, 
                                 cwd=os.path.join(self.out_dir, 'temp', 'Source'))
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, 'temp',
                                               'lib', 'libdhelas.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, 'temp',
                                               'lib', 'libmodel.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, 'temp',
                                               'lib', 'libgeneric.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, 'temp',
                                               'lib', 'libcernlib.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, 'temp',
                                               'lib', 'libdsample.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, 'temp',
                                               'lib', 'libpdf.a')))
        # Check that gensym compiles
        status = subprocess.call(['make', 'gensym'],
                                 stdout=devnull, stderr=devnull, 
                                 cwd=os.path.join(self.out_dir, 'temp', 'SubProcesses',
                                                  'P0_epem_epem'))
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, 'temp',
                                                    'SubProcesses',
                                                    'P0_epem_epem',
                                                    'gensym')))
        # Check that gensym runs
        proc = subprocess.Popen('./gensym', 
                                 stdout=devnull, stderr=devnull, stdin=subprocess.PIPE,
                                 cwd=os.path.join(self.out_dir, 'temp', 'SubProcesses',
                                                  'P0_epem_epem'), shell=True)
        proc.communicate('100 2 0.1 .false.\n'.encode())
        self.assertEqual(proc.returncode, 0)
        # Check that madevent compiles
        status = subprocess.call(['make', 'madevent'],
                                 stdout=devnull, stderr=devnull, 
                                 cwd=os.path.join(self.out_dir, 'temp', 'SubProcesses',
                                                  'P0_epem_epem'))
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, 'temp',
                                                    'SubProcesses',
                                                    'P0_epem_epem',
                                                    'madevent')))

    def test_invalid_operations_for_add(self):
        """Test that errors are raised appropriately for add"""

        self.assertRaises(InvalidCmd,
                          self.do, 'add process')
        self.assertRaises(InvalidCmd,
                          self.do, 'add wrong wrong')

    def test_invalid_operations_for_generate(self):
        """Test that errors are raised appropriately for generate"""

        self.assertRaises(MadGraph5Error,
                          self.do, 'generate')
        self.assertRaises(MadGraph5Error,
                          self.do, 'generate q q > q q')
        self.assertRaises(MadGraph5Error,
                          self.do, 'generate u u~ >')
        self.assertRaises(MadGraph5Error,
                          self.do, 'generate > u u~')
        self.assertRaises(MadGraph5Error,
                          self.do, 'generate a|z > b b~')

    def test_invalid_operations_for_output(self):
        """Test that errors are raised appropriately for output"""

        self.assertRaises(InvalidCmd,
                          self.do, 'output')

    def test_check_generate_optimize(self):
        """Test that errors are raised appropriately for output"""

        # Invalid since forbiddent by the optimize option
        self.assertRaises(InvalidCmd,
                          self.do, 'generate a > e+ e- --optimize')

        self.assertRaises(InvalidCmd,
                          self.do, 'generate b > t w- --optimize')

        # Invalid since optimize is not allowed for cross-section
        self.assertRaises(InvalidCmd,
                          self.do, 'generate  p p > e+ e- --optimize') 
        
        # check that --optimize filter correctly
        self.do('generate t > all all --optimize')
        self.assertEqual(len(self.cmd._curr_amps), 1)
              
               

    def test_read_madgraph4_proc_card(self):
        """Test reading a madgraph4 proc_card.dat"""
        os.system('cp -rf %s %s' % (os.path.join(MG5DIR,'Template','LO'),
                                    self.out_dir))
        os.system('cp -rf %s %s' % (
                            TestCmdShell1.join_path(_pickle_path,'simple_v4_proc_card.dat'),
                            os.path.join(self.out_dir,'Cards','proc_card.dat')))
    
        self.cmd = Cmd.MasterCmd()
        pwd = os.getcwd()
        os.chdir(self.out_dir)
        try:
            self.do('import proc_v4 %s' % os.path.join('Cards','proc_card.dat'))
        except:
            os.chdir(pwd)
            raise
        os.chdir(pwd)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                              'SubProcesses', 'P1_ll_vlvl')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                 'Cards', 'proc_card_mg5.dat')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P1_ll_vlvl',
                                                    'matrix1.ps')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'madevent.tar.gz')))
        


    def test_output_standalone_directory(self):
        """Test command 'output' with path"""
        
        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        self.do('set group_subprocesses False')
        self.do('import model_v4 sm')
        self.do('generate e+ e- > e+ e-')
        self.do('output standalone %s' % self.out_dir)
        self.do('set group_subprocesses True')
        self.assertTrue(os.path.exists(self.out_dir))
        self.assertTrue(os.path.isfile(os.path.join(self.out_dir, 'lib', 'libdhelas.a')))
        self.assertTrue(os.path.isfile(os.path.join(self.out_dir, 'lib', 'libmodel.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'SubProcesses', 'P0_epem_epem')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'Cards', 'proc_card_mg5.dat')))
    
    def test_custom_propa(self):
        """check that using custom propagator is working"""
        
        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        path = os.path.join(MG5DIR, 'tests', 'input_files', 'sm_with_custom_propa')
        self.do('import model %s' % path)
        self.do('generate g g > t t~')
        self.do('output standalone %s ' % self.out_dir)        
        
        files = ['aloha_file.inc', 'aloha_functions.f','FFV1_0.f', 'FFV1_1.f',
                 'FFV1_2.f', 'makefile', 'VVV1PV2_1.f'] 

        for f in files:
            self.assertTrue(os.path.isfile(os.path.join(self.out_dir,
                                                        'Source', 'DHELAS',
                                                        f)), 
                            '%s file is not in aloha directory' % f)

        devnull = open(os.devnull,'w')
        # Check that the Model and Aloha output compile
        subprocess.call(['make'],
                        stdout=devnull, stderr=devnull, 
                        cwd=os.path.join(self.out_dir, 'Source'))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libdhelas.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libmodel.a')))
        # Check that check_sa.f compiles
        subprocess.call(['make', 'check'],
                        stdout=devnull, stderr=devnull, 
                        cwd=os.path.join(self.out_dir, 'SubProcesses',
                                         'P0_gg_ttx'))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses', 'P0_gg_ttx',
                                                    'check')))
        # Check that the output of check is correct 
        logfile = os.path.join(self.out_dir,'SubProcesses', 'P0_gg_ttx',
                               'check.log')
        p = subprocess.Popen('./check', 
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        cwd=os.path.join(self.out_dir, 'SubProcesses',
                                         'P0_gg_ttx'), shell=True)
        (log_output, err) = p.communicate()
        log_output = log_output.decode()

        #log_output = open(logfile, 'r').read()
        #misc.sprint(log_output)
        me_re = re.compile(r'Matrix element\s*=\s*(?P<value>[\d\.eE\+-]+)\s*GeV',
                           re.IGNORECASE)
        me_groups = me_re.search(log_output)
        self.assertTrue(me_groups)
        self.assertAlmostEqual(float(me_groups.group('value')), 0.592626100)
        
    
    
    def test_ufo_aloha(self):
        """Test the import of models and the export of Helas Routine """

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        self.do('import model sm')
        self.do('generate e+ e- > e+ e-')
        self.do('output standalone %s ' % self.out_dir)
        # Check that the needed ALOHA subroutines are generated
        files = ['aloha_file.inc', 
                 #'FFS1C1_2.f', 'FFS1_0.f',
                 'FFV1_0.f', 'FFV1P0_3.f',
                 'FFV2_0.f', 'FFV2_3.f',
                 'FFV4_0.f', 'FFV4_3.f',
                 'makefile', 'aloha_functions.f']
        for f in files:
            self.assertTrue(os.path.isfile(os.path.join(self.out_dir,
                                                        'Source', 'DHELAS',
                                                        f)), 
                            '%s file is not in aloha directory' % f)
        # Check that unwanted ALOHA subroutines are not generated
        notfiles = ['FFV1_1.f', 'FFV1_2.f', 'FFV2_1.f', 'FFV2_2.f',
                    'FFV1_3.f','FFV2P0_3.f','FFV4P0_3.f'
                    'FFV4_1.f', 'FFV4_2.f', 
                    'VVV1_0.f', 'VVV1_1.f', 'VVV1_2.f', 'VVV1_3.f']
        for f in notfiles:
            self.assertFalse(os.path.isfile(os.path.join(self.out_dir,
                                                        'Source', 'DHELAS',
                                                        f)))
        devnull = open(os.devnull,'w')
        # Check that the Model and Aloha output compile
        subprocess.call(['make'],
                        stdout=devnull, stderr=devnull, 
                        cwd=os.path.join(self.out_dir, 'Source'))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libdhelas.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libmodel.a')))
        # Check that check_sa.f compiles
        subprocess.call(['make', 'check'],
                        stdout=devnull, stderr=devnull, 
                        cwd=os.path.join(self.out_dir, 'SubProcesses',
                                         'P0_epem_epem'))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses', 'P0_epem_epem',
                                                    'check')))
        # Check that the output of check is correct 
        logfile = os.path.join(self.out_dir,'SubProcesses', 'P0_epem_epem',
                               'check.log')
        p = subprocess.Popen('./check', 
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        cwd=os.path.join(self.out_dir, 'SubProcesses',
                                         'P0_epem_epem'), shell=True)
        (log_output, err) = p.communicate()
        log_output = log_output.decode()
        me_re = re.compile(r'Matrix element\s*=\s*(?P<value>[\d\.eE\+-]+)\s*GeV',
                           re.IGNORECASE)
        me_groups = me_re.search(log_output)
        self.assertTrue(me_groups)
        self.assertAlmostEqual(float(me_groups.group('value')), 1.953735e-2)
    
    def test_standalone_cpp(self):
        """test that standalone cpp is working"""

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        self.do('import model MSSM_SLHA2-full')
        self.do('generate g g > go go QED=2')
        self.do('output standalone_cpp %s ' % self.out_dir)
        devnull = open(os.devnull,'w')
    
        logfile = os.path.join(self.out_dir,'SubProcesses', 'P0_Sigma_MSSM_SLHA2_full_gg_gogo',
                               'check.log')
        # Check that check_sa.cc compiles
        subprocess.call(['make'],
                        stdout=devnull, stderr=devnull, 
                        cwd=os.path.join(self.out_dir, 'SubProcesses',
                                         'P0_Sigma_MSSM_SLHA2_full_gg_gogo'))
        
        subprocess.call('./check', 
                        stdout=open(logfile, 'w'), stderr=subprocess.STDOUT,
                        cwd=os.path.join(self.out_dir, 'SubProcesses',
                                         'P0_Sigma_MSSM_SLHA2_full_gg_gogo'), shell=True)
    
        log_output = open(logfile, 'r').read()
        me_re = re.compile(r'Matrix element\s*=\s*(?P<value>[\d\.eE\+-]+)\s*GeV',
                           re.IGNORECASE)
        me_groups = me_re.search(log_output)
        
        self.assertTrue(me_groups)
        self.assertAlmostEqual(float(me_groups.group('value')), 5.8183784340260782,5)
    
    
    def test_standalone_cpp_output_consistency(self):
        """test that standalone cpp is working"""

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        #step 0 cpp output
        self.do('generate p p > t t~, t > b mu+ vm, t~ > b~ mu- vm~')
        self.do('output standalone_cpp %s ' % self.out_dir)
        devnull = open(os.devnull,'w')
    
        directories= ['P0_Sigma_sm_gg_bmupvmbxmumvmx', 'P0_Sigma_sm_uux_bmupvmbxmumvmx']
        def get_values():
            values = []
            for oneproc in directories:
                logfile = os.path.join(self.out_dir,'SubProcesses', oneproc,
                                       'check.log')
                # Check that check_sa.cc compiles
                subprocess.call(['make'],
                                stdout=devnull, stderr=devnull, 
                                cwd=os.path.join(self.out_dir, 'SubProcesses', oneproc))
                
                subprocess.call('./check', 
                                stdout=open(logfile, 'w'), stderr=subprocess.STDOUT,
                                cwd=os.path.join(self.out_dir, 'SubProcesses',
                                                 oneproc), shell=True)
            
                log_output = open(logfile, 'r').read()
                me_re = re.compile(r'Matrix element\s*=\s*(?P<value>[\d\.eE\+-]+)\s*GeV',
                                   re.IGNORECASE)
                me_groups = me_re.search(log_output)
                self.assertTrue(me_groups)
                values.append(float(me_groups.group('value')))
            return values
        original = get_values()
        #step 1 standalone output
        shutil.rmtree(self.out_dir)
        self.do('output standalone %s -f' % self.out_dir)
        shutil.rmtree(self.out_dir)            
        self.do('output standalone_cpp %s -f' % self.out_dir)     
        new = get_values()
        
        for i,_ in enumerate(original):
            self.assertEqual(original[i], new[i])

         
    def test_standalone_density(self):
        """test that standalone density is working"""

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        self.do('generate p p > j t t~ ')
        self.do('output standalone %s --density=4,5 -f' % self.out_dir)
        devnull = open(os.devnull,'w')
    
        logfile = os.path.join(self.out_dir,'SubProcesses', 'P0_gg_gttx',
                               'check_sa.log')
        # Check that check_sa.f compiles
        subprocess.call(['make'],
                        stdout=devnull, stderr=devnull, 
                        cwd=os.path.join(self.out_dir, 'SubProcesses',
                                         'P0_gg_gttx'))
        
        subprocess.call(['./check'],
                        stdout=open(logfile, 'w'), stderr=subprocess.STDOUT,
                        cwd=os.path.join(self.out_dir, 'SubProcesses',
                                         'P0_gg_gttx'))
    
        log_output = open(logfile, 'r').read()
        me_re = re.compile(r'Matrix element\s*=\s*(?P<value>[\d\.eE\+-]+)\s*GeV',
                           re.IGNORECASE)
        me_groups = me_re.search(log_output)    
        self.assertTrue(me_groups)
        self.assertAlmostEqual(float(me_groups.group('value')), 0.0004188716594531423, 5)

        # check density matrix
        #
        #particle 3 has helicity -1 -1
        #particle 4 has helicity 1 1
        #value is 1 (2.78194898230116219E-011,0.0000000000000000)
        me_re = re.compile(r'particle\s+4\s+has\s+helicity\s+([\+\-]?\d)\s*([\+\-]?\d)\s*'
                           r'particle\s+5\s+has\s+helicity\s+([\+\-]?\d)\s*([\+\-]?\d)\s*'
                           r'value is\s+\d+\s*\(([\d\.eE\+-]+),([\d\.eE\+-]+)\)', re.MULTILINE)
        all_matches = me_re.findall(log_output)
        sol = {}
        for match in all_matches:
            sol[(int(match[0]), int(match[1]), int(match[2]), int(match[3]))] = (float(match[4]), float(match[5]))

        original_sol = {(-1, -1, 1, 1): (0.02827952274928987, 0.0), (-1, -1, 1, -1): (-0.0041892876162345, -0.0041923830983622255), (-1, 1, 1, 1): (0.000469685615962711, 0.0006142055733429721), (-1, 1, 1, -1): (-0.01784029173125566, -0.00794999696313525), (-1, -1, -1, -1): (0.02532739017396033, 0.0), (-1, 1, -1, 1): (-0.00028182588524174187, 0.0024162264334765746), (-1, 1, -1, -1): (-0.00048593945847553023, -0.0006039982074415239), (1, 1, 1, 1): (0.025301510150454294, 0.0), (1, 1, 1, -1): (0.004212401136919661, 0.0042167644618831875), (1, 1, -1, -1): (0.028322721746299958, 0.0)}

        for key in original_sol:
            self.assertIn(key, sol)
            self.assertAlmostEqual(original_sol[key][0], sol[key][0])
            self.assertAlmostEqual(original_sol[key][1], sol[key][1])

        ########################################################################
        ### check case with polarization vectors
        ########################################################################
        self.do('generate u u~ > z{0} z{T} g')
        self.do('output standalone %s --density=3,4,5 -f ' % self.out_dir)
        devnull = open(os.devnull,'w')
    
        logfile = os.path.join(self.out_dir,'SubProcesses', 'P0_uux_z0zTg',
                               'check_sa.log')
        # Check that check_sa.f compiles
        subprocess.call(['make'],
                        stdout=devnull, stderr=devnull, 
                        cwd=os.path.join(self.out_dir, 'SubProcesses',
                                         'P0_uux_z0zTg'))
        
        subprocess.call(['./check'],
                        stdout=open(logfile, 'w'), stderr=subprocess.STDOUT,
                        cwd=os.path.join(self.out_dir, 'SubProcesses',
                                         'P0_uux_z0zTg'))
    
        log_output = open(logfile, 'r').read()
        me_re = re.compile(r'Matrix element\s*=\s*(?P<value>[\d\.eE\+-]+)\s*GeV',
                           re.IGNORECASE)
        me_groups = me_re.search(log_output)    
        self.assertTrue(me_groups)
        self.assertAlmostEqual(float(me_groups.group('value')), 3.140334500813846e-08, 5)

        # check density matrix
        #
        #particle 3 has helicity -1 -1
        #particle 4 has helicity 1 1
        #value is 1 (2.78194898230116219E-011,0.0000000000000000)
        me_re = re.compile(r'particle\s+3\s+has\s+helicity\s+([\+\-]?\d)\s*([\+\-]?\d)\s*'
                           r'particle\s+4\s+has\s+helicity\s+([\+\-]?\d)\s*([\+\-]?\d)\s*'
                           r'particle\s+5\s+has\s+helicity\s+([\+\-]?\d)\s*([\+\-]?\d)\s*'
                           r'value is\s+\d+\s*\(([\d\.eE\+-]+),([\d\.eE\+-]+)\)', re.MULTILINE)
        all_matches = me_re.findall(log_output)
        sol = {}
        for match in all_matches:
            sol[(int(match[0]), int(match[1]), int(match[2]), int(match[3]), int(match[4]),int(match[5]))] = (float(match[6]), float(match[7]))
            self.assertTrue(len(match), 8)
            self.assertEqual(int(match[0]), int(match[1]))
            self.assertEqual(int(match[0]), 0)
            self.assertIn(int(match[2]), [-1,1])
            self.assertIn(int(match[3]), [-1,1])

        self.assertEqual(len(sol), 10)
        original_sol =  {(0, 0, 1, 1, -1, -1): (3.4001167694559294e-07, 0.0), (0, 0, 1, 1, -1, 1): (-3.1461674759236455e-07, 4.8545818497513406e-09), (0, 0, 1, -1, -1, -1): (-3.069602710730949e-07, 4.832171908212019e-09), (0, 0, 1, -1, -1, 1): (2.7270589345996334e-07, -2.0762612868036477e-08), (0, 0, 1, 1, 1, 1): (2.914827381801457e-07, 0.0), (0, 0, 1, -1, 1, -1): (2.8446952890247865e-07, -7.217171184019204e-11), (0, 0, 1, -1, 1, 1): (-2.5326248092821595e-07, 1.519703606636158e-08), (0, 0, -1, -1, -1, -1): (2.7764709546297174e-07, 0.0), (0, 0, -1, -1, -1, 1): (-2.4727978606009926e-07, 1.4752928639145769e-08), (0, 0, -1, -1, 1, 1): (2.2137890970427402e-07, 0.0)}
        for key in original_sol:
            
            self.assertIn(key, sol)
            self.assertAlmostEqual(original_sol[key][0], sol[key][0])
            self.assertAlmostEqual(original_sol[key][1], sol[key][1])

        ########################################################################
        ### check Z > t t~ case
        ######################################################################## 
        self.do('generate z > b b~')
        self.do('output standalone %s --density=1 -f ' % self.out_dir)
        devnull = open(os.devnull,'w')
    
        logfile = os.path.join(self.out_dir,'SubProcesses', 'P0_z_bbx',
                               'check_sa.log')
        # Check that check_sa.f compiles
        subprocess.call(['make'],
                        stdout=devnull, stderr=devnull, 
                        cwd=os.path.join(self.out_dir, 'SubProcesses',
                                         'P0_z_bbx'))
        
        subprocess.call(['./check'],
                        stdout=open(logfile, 'w'), stderr=subprocess.STDOUT,
                        cwd=os.path.join(self.out_dir, 'SubProcesses',
                                         'P0_z_bbx'))
    
        log_output = open(logfile, 'r').read()
        me_re = re.compile(r'Matrix element\s*=\s*(?P<value>[\d\.eE\+-]+)\s*GeV',
                           re.IGNORECASE)
        me_groups = me_re.search(log_output) 
        self.assertTrue(me_groups)
        self.assertAlmostEqual(float(me_groups.group('value')), 1696.1593018147787, 5)

        # check density matrix
        #
        #particle 3 has helicity -1 -1
        #particle 4 has helicity 1 1
        #value is 1 (2.78194898230116219E-011,0.0000000000000000)
        me_re = re.compile(
                           r'particle\s+1\s+has\s+helicity\s+([\+\-]?\d)\s*([\+\-]?\d)\s*'
                           r'value is\s+\d+\s*\(([\d\.eE\+-]+),([\d\.eE\+-]+)\)', re.MULTILINE)
        all_matches = me_re.findall(log_output)
        sol = {}
        for match in all_matches:
            sol[(int(match[0]), int(match[1]))] = (float(match[2]), float(match[3]))


        self.assertEqual(len(sol), 6)
        original_sol =  {(-1, -1): (520.1204099513759, 0.0), (-1, 0): (-217.2395066557355, 871.1782252029083), (-1, 1): (-939.258737149777, -499.49184104989456), (0, 0): (2136.628541206853, 0.0), (0, 1): (-534.1281427839928, 2141.9713873632077), (1, 1): (2431.7289542861076, 0.0)}
        for key in original_sol:
            
            self.assertIn(key, sol)
            self.assertAlmostEqual(original_sol[key][0], sol[key][0])
            self.assertAlmostEqual(original_sol[key][1], sol[key][1])


        ########################################################################
        ### check case with interference computation
        ######################################################################## 
        self.do('generate u u~ > t t~ QCD^2==2')
        self.do('output standalone %s --density=3,4 -f ' % self.out_dir)
        devnull = open(os.devnull,'w')
    
        logfile = os.path.join(self.out_dir,'SubProcesses', 'P0_uux_ttx',
                               'check_sa.log')
        # Check that check_sa.f compiles
        subprocess.call(['make'],
                        stdout=devnull, stderr=devnull, 
                        cwd=os.path.join(self.out_dir, 'SubProcesses',
                                         'P0_uux_ttx'))
        
        subprocess.call(['./check'],
                        stdout=open(logfile, 'w'), stderr=subprocess.STDOUT,
                        cwd=os.path.join(self.out_dir, 'SubProcesses',
                                         'P0_uux_ttx'))
    
        log_output = open(logfile, 'r').read()
        me_re = re.compile(r'Matrix element\s*=\s*(?P<value>[\d\.eE\+-]+)\s*GeV',
                           re.IGNORECASE)
        me_groups = me_re.search(log_output) 
        self.assertTrue(me_groups)
        self.assertAlmostEqual(float(me_groups.group('value')), 0, 5)

        # check density matrix
        #
        #particle 3 has helicity -1 -1
        #particle 4 has helicity 1 1
        #value is 1 (2.78194898230116219E-011,0.0000000000000000)
        me_re = re.compile(
                           r'particle\s+3\s+has\s+helicity\s+([\+\-]?\d)\s*([\+\-]?\d)\s*'
                           r'particle\s+4\s+has\s+helicity\s+([\+\-]?\d)\s*([\+\-]?\d)\s*'
                           r'value is\s+\d+\s*\(([\d\.eE\+-]+),([\d\.eE\+-]+)\)', re.MULTILINE)
        all_matches = me_re.findall(log_output)
        sol = {}
        for match in all_matches:
            sol[(int(match[0]), int(match[1]), int(match[2]), int(match[3]))] = (float(match[4]), float(match[5]))
            self.assertTrue(len(match), 6)
            self.assertIn(int(match[0]), [-1,1])
            self.assertIn(int(match[1]), [-1,1])
            self.assertIn(int(match[2]), [-1,1])
            self.assertIn(int(match[3]), [-1,1])


        self.assertEqual(len(sol), 10)
        original_sol = sol =  {(-1, -1, 1, 1): (-5.551115123125783e-17, 0.0), (-1, -1, 1, -1): (0.00926133766116009, 0.0), (-1, 1, 1, 1): (-0.009292545047075378, 0.0), (-1, 1, 1, -1): (0.09884700885130696, 0.0), (-1, -1, -1, -1): (-1.6479873021779667e-17, 0.0), (-1, 1, -1, 1): (5.415404411525035e-07, 0.0), (-1, 1, -1, -1): (-0.012877349160785171, 0.0), (1, 1, 1, 1): (-1.734723475976807e-17, 0.0), (1, 1, 1, -1): (0.012880402732054583, 0.0), (1, 1, -1, -1): (-4.85722573273506e-17, 0.0)}
        for key in original_sol:

            self.assertIn(key, sol)
            self.assertAlmostEqual(original_sol[key][0] if abs(original_sol[key][0]) > 1e-12 else 0, sol[key][0])
            self.assertAlmostEqual(original_sol[key][1] if abs(original_sol[key][1]) > 1e-12 else 0, sol[key][1])

    def test_standalone_density_tmp(self):
        ############################################################################
        # Check convolution of density matrix with decay matrix 
        # reproduces the full matrix-element
        # testing case u u~ > z z, z > e+ e-
        ############################################################################
        self.do('generate u u~ > z z')
        self.do('output standalone %s_prod --density=3,4 -f ' % self.out_dir)
        self.do('generate u u~ > z z, z > e+ e-')
        self.do('output standalone %s_full -f ' % self.out_dir) 
        self.do('generate z > e+ e- --standalone') # --standalone allow mix 2>1 and 2>2 processes
        self.do('output standalone %s_dec1 --density=1 -f ' % self.out_dir)
        self.do('output standalone %s_dec2 --density=1 -f ' % self.out_dir)
        # Read a test event for u u~ > z z, z > e+ e-
        text_lhe = """<event>
          8      1 +9.3742000e+00 9.47120700e+01 7.54677100e-03 1.29182600e-01
        -1 -1    0    0    0  501 -0.0000000000e+00 +0.0000000000e+00 +2.9573272973e+01 2.9573272973e+01 0.0000000000e+00 0.0000e+00 1.0000e+00
        1 -1    0    0  501    0 +0.0000000000e+00 -0.0000000000e+00 -3.1517472072e+02 3.1517472072e+02 0.0000000000e+00 0.0000e+00 -1.0000e+00
       23  2    1    2    0    0 -2.1281935632e+01 +1.4219874445e+01 -1.0938041178e+02 1.4468742668e+02 9.1188000000e+01 0.0000e+00 0.0000e+00
       23  2    1    2    0    0 +2.1281935632e+01 -1.4219874445e+01 -1.7622103597e+02 2.0006056702e+02 9.1188000000e+01 0.0000e+00 0.0000e+00
      -11  1    3    3    0    0 -2.3913658500e+01 -2.5175464738e+01 -7.6385381990e+00 3.5552979530e+01 0.0000000000e+00 0.0000e+00 1.0000e+00
       11  1    3    3    0    0 +2.6317228690e+00 +3.9395339183e+01 -1.0174187358e+02 1.0913444714e+02 0.0000000000e+00 0.0000e+00 -1.0000e+00
      -11  1    4    4    0    0 -5.3058986557e+00 -4.9901564188e+01 -9.5949709558e+01 1.0828049424e+02 0.0000000000e+00 0.0000e+00 -1.0000e+00
       11  1    4    4    0    0 +2.6587834287e+01 +3.5681689743e+01 -8.0271326406e+01 9.1780072772e+01 0.0000000000e+00 0.0000e+00 1.0000e+00
       </event>"""


        
        Event = lhe_parser.Event()
        Event.parse(text_lhe.split('\n')) 
        # get the associate momenta for each matrix-element
        production_p = [Event[0], Event[1], Event[2], Event[3]]
        full_p = [Event[0], Event[1], Event[4], Event[5], Event[6], Event[7]]
        decay_z1 = [Event[2], Event[4], Event[5]]
        decay_z2 = [Event[3], Event[6], Event[7]]
        all_p = [production_p, full_p, decay_z1, decay_z2]


        all_me = []
        all_dens = []
        all_index= []
        # compiles the three directories:
        for i, (mdir, Pdir) in enumerate([(self.out_dir+'_prod', 'P0_uux_zz'), 
                                 (self.out_dir+'_full', 'P0_uux_zz_z_epem_z_epem'), 
                                 (self.out_dir+'_dec1', 'P0_z_epem'),
                                 (self.out_dir+'_dec2', 'P0_z_epem')]):
            p = all_p[i]
            self.edit_p_in_standalone(os.path.join(mdir, 'SubProcesses', Pdir), p)
            devnull = open(os.devnull,'w')
            subprocess.call(['make'],
                            stdout=devnull, stderr=devnull, 
                            cwd=os.path.join(mdir, 'SubProcesses',
                                             Pdir)) 
            self.assertTrue(os.path.exists(os.path.join(mdir,
                                                        'SubProcesses', Pdir,
                                                        'check')))
            #compute the matrix-element
            logfile = os.path.join(mdir,'SubProcesses', Pdir,
                                   'check.log')
            subprocess.call('./check', 
                            stdout=open(logfile, 'w'), stderr=subprocess.STDOUT,
                            cwd=os.path.join(mdir, 'SubProcesses',
                                             Pdir), shell=True)
            log_output = open(logfile, 'r').read()
            misc.sprint(log_output)
            me_re = re.compile(r'Matrix element\s*=\s*(?P<value>[\d\.eE\+-]+)\s*GeV',
                               re.IGNORECASE)
            me_groups = me_re.findall(log_output) 
            dens_re =re.compile(r'value is\s+\d+\s*\(([\d\.eE\+-]+),([\d\.eE\+-]+)\)')
            density = dens_re.findall(log_output)

            hel_index = re.compile(
                           r'particle\s+\d\s+has\s+helicity\s+([\+\-]?\d)\s*([\+\-]?\d)\s*')
            index = hel_index.findall(log_output)
    
            if len(index) == len(density):
                all_index.append([[int(x[0]), int(x[1])] for x in index])
            else:
                curr_index = []
                for i in range(len(all_p), 2):
                    i1 = index[2*i]
                    i2 = index[2*i+1]
                    curr_index.append([[int(i1[0]), int(i1[1]), int(i2[0]), int(i2[1])]])
              
                all_index.append(curr_index)

            
            all_me.append(float(me_groups[0]))
            all_dens.append([complex(float(x[0]), float(x[1])) for x in density])

        import MadSpin.decay as madspin
        self.assertEqual(all_dens[1], []) 

        import numpy as np
        # array, nchanging, all_helicity_combinations, dimension):
        allow_hel = [-1,-1,   -1,0,   -1,1, 
                        0,-1,  0,0,   0,1, 
                        1,-1,  1,0,   1,1]
        
        #allow_hel = [-1,-1,-1,0,0,0,1,1,1,
        #             -1,0,1,-1,0,1,-1,0,1]

        misc.sprint(all_index[0])
        prod_dens = madspin.DensityMatrix(all_dens[0], 2, allow_hel, 9) 
        prod_dec1 = madspin.DensityMatrix(all_dens[2], 1, [-1,0,1], 3) 
        prod_dec2 = madspin.DensityMatrix(all_dens[3], 1, [-1,0,1], 3)  

        self.assertAlmostEqual(prod_dec1.trace()/3., all_me[2],4)
        self.assertAlmostEqual(prod_dec2.trace()/3., all_me[3],4)
        self.assertAlmostEqual(prod_dens.trace()/9./4./2., all_me[0],4)  #9 color , 4 spin, 2 symmetry factor (ZZ)


        prod_dec =prod_dec1.tensor_product(prod_dec2)
        prod_dec_sym =prod_dec2.tensor_product(prod_dec1) 
        mZ= 91.18800
        WZ = 2.44140
        nb_hel = 3*3
        symfact = 2 # 2 Z identical particles in the final state
        nb_spin = 2*2
        matrix = prod_dens.scalar_multiplication(prod_dec)/mZ**4/WZ**4/nb_hel/symfact/nb_spin
        matrix_sym = prod_dens.scalar_multiplication(prod_dec_sym)/mZ**4/WZ**4/nb_hel/symfact/nb_spin 

        misc.sprint(matrix/all_me[1], all_me[1]/matrix)
        misc.sprint(matrix_sym/all_me[1], all_me[1]/matrix_sym) 
        self.assertAlmostEqual(matrix, all_me[1],4)




   
        ############################################################################
        # Check convolution of density matrix with decay matrix 
        # reproduces the full matrix-element
        # testing case u u~ > z z, z > e+ e-
        ############################################################################
        self.do('generate d d~ > z z')
        self.do('output standalone %s_prod --density=3,4 -f ' % self.out_dir)
        self.do('generate d d~ > z z, z > e+ e-')
        self.do('output standalone %s_full -f ' % self.out_dir) 
        self.do('generate z > e+ e- --standalone') # --standalone allow mix 2>1 and 2>2 processes
        self.do('output standalone %s_dec1 --density=1 -f ' % self.out_dir)
        self.do('output standalone %s_dec2 --density=1 -f ' % self.out_dir)
        # Read a test event for u u~ > z z, z > e+ e-
        text_lhe = """ <event>
 8      1 +9.3182000e+00 9.58106800e+01 7.54677100e-03 1.28936100e-01
       -2 -1    0    0    0  501 -0.0000000000e+00 +0.0000000000e+00 +1.3133897947e+02 1.3133897947e+02 0.0000000000e+00 0.0000e+00 1.0000e+00
        2 -1    0    0  501    0 +0.0000000000e+00 -0.0000000000e+00 -1.8928771624e+02 1.8928771624e+02 0.0000000000e+00 0.0000e+00 -1.0000e+00
       23  2    1    2    0    0 +2.7856507549e+01 +9.4048216813e+00 -1.5629560828e+02 1.8332485973e+02 9.1188000000e+01 0.0000e+00 0.0000e+00
       23  2    1    2    0    0 -2.7856507549e+01 -9.4048216813e+00 +9.8346871509e+01 1.3730183598e+02 9.1188000000e+01 0.0000e+00 0.0000e+00
      -11  1    3    3    0    0 +4.1207329404e+01 +1.3210392522e+01 -2.2428975695e+01 4.8740305887e+01 0.0000000000e+00 0.0000e+00 1.0000e+00
       11  1    3    3    0    0 -1.3350821855e+01 -3.8055708401e+00 -1.3386663259e+02 1.3458455385e+02 0.0000000000e+00 0.0000e+00 -1.0000e+00
      -11  1    4    4    0    0 -5.7647128134e+01 -2.1962673907e+01 +7.6540783391e+01 9.8305859181e+01 0.0000000000e+00 0.0000e+00 -1.0000e+00
       11  1    4    4    0    0 +2.9790620585e+01 +1.2557852226e+01 +2.1806088117e+01 3.8995976798e+01 0.0000000000e+00 0.0000e+00 1.0000e+00
       </event>"""


        
        Event = lhe_parser.Event()
        Event.parse(text_lhe.split('\n')) 
        # get the associate momenta for each matrix-element
        production_p = [Event[0], Event[1], Event[2], Event[3]]
        full_p = [Event[0], Event[1], Event[4], Event[5], Event[6], Event[7]]
        decay_z1 = [Event[2], Event[4], Event[5]]
        decay_z2 = [Event[3], Event[6], Event[7]]
        all_p = [production_p, full_p, decay_z1, decay_z2]


        all_me = []
        all_dens = []
        all_index= []
        # compiles the three directories:
        for i, (mdir, Pdir) in enumerate([(self.out_dir+'_prod', 'P0_ddx_zz'), 
                                 (self.out_dir+'_full', 'P0_ddx_zz_z_epem_z_epem'), 
                                 (self.out_dir+'_dec1', 'P0_z_epem'),
                                 (self.out_dir+'_dec2', 'P0_z_epem')]):
            p = all_p[i]
            self.edit_p_in_standalone(os.path.join(mdir, 'SubProcesses', Pdir), p)
            devnull = open(os.devnull,'w')
            subprocess.call(['make'],
                            stdout=devnull, stderr=devnull, 
                            cwd=os.path.join(mdir, 'SubProcesses',
                                             Pdir)) 
            self.assertTrue(os.path.exists(os.path.join(mdir,
                                                        'SubProcesses', Pdir,
                                                        'check')))
            #compute the matrix-element
            logfile = os.path.join(mdir,'SubProcesses', Pdir,
                                   'check.log')
            subprocess.call('./check', 
                            stdout=open(logfile, 'w'), stderr=subprocess.STDOUT,
                            cwd=os.path.join(mdir, 'SubProcesses',
                                             Pdir), shell=True)
            log_output = open(logfile, 'r').read()
            misc.sprint(log_output)
            me_re = re.compile(r'Matrix element\s*=\s*(?P<value>[\d\.eE\+-]+)\s*GeV',
                               re.IGNORECASE)
            me_groups = me_re.findall(log_output) 
            dens_re =re.compile(r'value is\s+\d+\s*\(([\d\.eE\+-]+),([\d\.eE\+-]+)\)')
            density = dens_re.findall(log_output)

            hel_index = re.compile(
                           r'particle\s+\d\s+has\s+helicity\s+([\+\-]?\d)\s*([\+\-]?\d)\s*')
            index = hel_index.findall(log_output)
    
            if len(index) == len(density):
                all_index.append([[int(x[0]), int(x[1])] for x in index])
            else:
                curr_index = []
                for i in range(len(all_p), 2):
                    i1 = index[2*i]
                    i2 = index[2*i+1]
                    curr_index.append([[int(i1[0]), int(i1[1]), int(i2[0]), int(i2[1])]])
              
                all_index.append(curr_index)

            
            all_me.append(float(me_groups[0]))
            all_dens.append([complex(float(x[0]), float(x[1])) for x in density])

        import MadSpin.decay as madspin
        self.assertEqual(all_dens[1], []) 

        import numpy as np
        # array, nchanging, all_helicity_combinations, dimension):
        allow_hel = [-1,-1,   -1,0,   -1,1, 
                      0,-1,    0,0,    0,1, 
                      1,-1,    1,0,    1,1] #double checked with fortran code.
        

        misc.sprint(all_index[0])
        prod_dens = madspin.DensityMatrix(all_dens[0], 2, allow_hel, 9) 
        prod_dec1 = madspin.DensityMatrix(all_dens[2], 1, [-1,0,1], 3) 
        prod_dec2 = madspin.DensityMatrix(all_dens[3], 1, [-1,0,1], 3)


        #consistency of the matrix-element and the density matrix

        self.assertAlmostEqual(prod_dec1.trace()/3., all_me[2],4)
        self.assertAlmostEqual(prod_dec2.trace()/3., all_me[3],4)
        self.assertAlmostEqual(prod_dens.trace()/9./4./2., all_me[0],4)  #9 color , 4 spin, 2 symmetry factor (ZZ)

        prod_dec =prod_dec1.tensor_product(prod_dec2)
        prod_dec_sym =prod_dec2.tensor_product(prod_dec1) 
        mZ= 91.18800
        WZ = 2.44140
        nb_hel = 3*3
        symfact = 2 # 2 Z identical particles in the final state
        nb_spin = 2*2
        matrix = prod_dens.scalar_multiplication(prod_dec)/mZ**4/WZ**4/nb_hel/symfact/nb_spin
        #matrix_sym = prod_dens.scalar_multiplication(prod_dec_sym)/mZ**4/WZ**4/nb_hel/symfact/nb_spin

        #misc.sprint(matrix, matrix_sym, all_me[1])
        #misc.sprint(matrix/all_me[1], matrix_sym/all_me[1])


        self.assertAlmostEqual(matrix, all_me[1],4)
        #self.assertAlmostEqual(matrix_sym, all_me[1],4)




    def edit_p_in_standalone(self, dir, p):
        """edit the check.f file to include the momenta p"""
        misc.sprint(dir)
        text = []
        for line in open(os.path.join(dir, 'check_sa.f'), 'r'):
            if 'CALL GET_MOMENTA(SQRTS,PMASS,P)' in line:
                for i in range(len(p)):
                    real_p = lhe_parser.FourMomentum(p[i])
                    misc.sprint(real_p)
                    for j in range(4):
                        text.append('        p(%s,%s) = %e\n' % (j, i+1, real_p[j]))
            else:
                text.append(line)


        checkf = os.path.join(dir, 'check_sa.f')
        open(checkf, 'w').write('\n'.join(text))

        
    def test_v4_heft(self):
        """Test standalone directory for UFO HEFT model"""

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        self.do('import model_v4 heft', force=True)
        self.do('generate g g > h g g')
        self.do('output standalone %s ' % self.out_dir)

        devnull = open(os.devnull,'w')
        # Check that the Model and Aloha output compile
        subprocess.call(['make'],
                        stdout=devnull, stderr=devnull, 
                        cwd=os.path.join(self.out_dir, 'Source'))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libdhelas.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libmodel.a')))
        # Check that check_sa.f compiles
        subprocess.call(['make', 'check'],
#                        stdout=devnull, stderr=devnull, 
                        cwd=os.path.join(self.out_dir, 'SubProcesses',
                                         'P0_gg_hgg'))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses', 'P0_gg_hgg',
                                                    'check')))
        # Check that the output of check is correct 
        logfile = os.path.join(self.out_dir,'SubProcesses', 'P0_gg_hgg',
                               'check.log')
        p = subprocess.Popen('./check', 
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        cwd=os.path.join(self.out_dir, 'SubProcesses',
                                         'P0_gg_hgg'), shell=True)
        (log_output, err) = p.communicate()                                         
        log_output =log_output.decode()
        me_re = re.compile(r'Matrix element\s*=\s*(?P<value>[\d\.eE\+-]+)\s*GeV',
                           re.IGNORECASE)
        me_groups = me_re.search(log_output)
        
        self.assertTrue(me_groups)
        self.assertAlmostEqual(float(me_groups.group('value')), 1.10908942e-06)
        
    def test_madevent_ufo_aloha(self):
        """Test MadEvent output with UFO/ALOHA"""

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        self.do('import model sm')
        self.do('set group_subprocesses False')
        self.do('generate e+ e- > e+ e-')
        self.do('output %s ' % self.out_dir)
        # Check that the needed ALOHA subroutines are generated
        files = ['aloha_file.inc', 
                 #'FFS1C1_2.f', 'FFS1_0.f',
                 'FFV1_0.f', 'FFV1P0_3.f',
                 'FFV2_0.f', 'FFV2_3.f',
                 'FFV4_0.f', 'FFV4_3.f',
                 'makefile', 'aloha_functions.f']
        for f in files:
            self.assertTrue(os.path.isfile(os.path.join(self.out_dir,
                                                        'Source', 'DHELAS',
                                                        f)), 
                            '%s file is not in aloha directory' % f)
        
        #check the content of FFV1P0_0.f
        self.check_aloha_file()
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'Cards',
                                                    'ident_card.dat')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                 'Cards', 'run_card_default.dat')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                 'Cards', 'plot_card_default.dat')))
        devnull = open(os.devnull,'w')
        # Check that the Source directory compiles
        status = subprocess.call(['make'],
                                 stdout=devnull, 
                                 cwd=os.path.join(self.out_dir, 'Source'))
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libdhelas.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libmodel.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libgeneric.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libcernlib.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libdsample.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libpdf.a')))
        # Check that gensym compiles
        status = subprocess.call(['make', 'gensym'],
                                 stdout=devnull, 
                                 cwd=os.path.join(self.out_dir, 'SubProcesses',
                                                  'P0_epem_epem'))
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P0_epem_epem',
                                                    'gensym')))
        # Check that gensym runs
        proc = subprocess.Popen('./gensym', 
                                 stdout=devnull, stdin=subprocess.PIPE,
                                 cwd=os.path.join(self.out_dir, 'SubProcesses',
                                                  'P0_epem_epem'), shell=True)
        proc.communicate('100 2 0.1 .false.\n'.encode())
        
        self.assertEqual(proc.returncode, 0)
        # Check that madevent compiles
        status = subprocess.call(['make', 'madevent'],
                                 stdout=devnull, 
                                 cwd=os.path.join(self.out_dir, 'SubProcesses',
                                                  'P0_epem_epem'))
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P0_epem_epem',
                                                    'madevent')))
        
        
    def check_aloha_file(self):
        """check the content of aloha file FFV1P0_3.f and FFV2_3.f"""
        
        ffv1p0 = """C     This File is Automatically generated by ALOHA 
C     The process calculated in this file is: 
C     Gamma(3,2,1)
C    
      SUBROUTINE FFV1P0_3(F1, F2, COUP, M3, W3,V3)
      IMPLICIT NONE
      COMPLEX*16 CI
      PARAMETER (CI=(0D0,1D0))
      COMPLEX*16 COUP
      COMPLEX*16 F1(*)
      COMPLEX*16 F2(*)
      REAL*8 M3
      REAL*8 P3(0:3)
      COMPLEX*16 V3(6)
      REAL*8 W3
      COMPLEX*16 DENOM
      V3(1) = +F1(1)+F2(1)
      V3(2) = +F1(2)+F2(2)
      P3(0) = -DBLE(V3(1))
      P3(1) = -DBLE(V3(2))
      P3(2) = -DIMAG(V3(2))
      P3(3) = -DIMAG(V3(1))
      DENOM = COUP/(P3(0)**2-P3(1)**2-P3(2)**2-P3(3)**2 - M3 * (M3 -CI
     $ * W3))
      V3(3)= DENOM*(-CI)*(F1(3)*F2(5)+F1(4)*F2(6)+F1(5)*F2(3)+F1(6)
     $ *F2(4))
      V3(4)= DENOM*(-CI)*(-F1(3)*F2(6)-F1(4)*F2(5)+F1(5)*F2(4)+F1(6)
     $ *F2(3))
      V3(5)= DENOM*(-CI)*(-CI*(F1(3)*F2(6)+F1(6)*F2(3))+CI*(F1(4)*F2(5)
     $ +F1(5)*F2(4)))
      V3(6)= DENOM*(-CI)*(-F1(3)*F2(5)-F1(6)*F2(4)+F1(4)*F2(6)+F1(5)
     $ *F2(3))
      END


"""
        text = open(os.path.join(self.out_dir,'Source', 'DHELAS', 'FFV1P0_3.f')).read()
        
        self.assertNotIn('OM3', text)
        ffv1p0 = [l.strip() for l in ffv1p0.strip().split('\n')]
        text = [l.strip() for l in text.strip().split('\n')]
        self.assertEqual(ffv1p0, text)
        
        ffv2 = """C     This File is Automatically generated by ALOHA 
C     The process calculated in this file is: 
C     Gamma(3,2,-1)*ProjM(-1,1)
C       
      SUBROUTINE FFV2_3(F1, F2, COUP, M3, W3,V3)
      IMPLICIT NONE
      COMPLEX*16 CI
      PARAMETER (CI=(0D0,1D0))
      COMPLEX*16 COUP
      COMPLEX*16 F1(*)
      COMPLEX*16 F2(*)
      REAL*8 M3
      REAL*8 OM3
      REAL*8 P3(0:3)
      COMPLEX*16 TMP2
      COMPLEX*16 V3(6)
      REAL*8 W3
      COMPLEX*16 DENOM
      OM3 = 0D0
      IF (M3.NE.0D0) OM3=1D0/M3**2
      V3(1) = +F1(1)+F2(1)
      V3(2) = +F1(2)+F2(2)
      P3(0) = -DBLE(V3(1))
      P3(1) = -DBLE(V3(2))
      P3(2) = -DIMAG(V3(2))
      P3(3) = -DIMAG(V3(1))
      TMP2 = (F1(3)*(F2(5)*(P3(0)+P3(3))+F2(6)*(P3(1)+CI*(P3(2))))
     $ +F1(4)*(F2(5)*(P3(1)-CI*(P3(2)))+F2(6)*(P3(0)-P3(3))))
      DENOM = COUP/(P3(0)**2-P3(1)**2-P3(2)**2-P3(3)**2 - M3 * (M3 -CI
     $ * W3))
      V3(3)= DENOM*(-CI)*(F1(3)*F2(5)+F1(4)*F2(6)-P3(0)*OM3*TMP2)
      V3(4)= DENOM*(-CI)*(-F1(3)*F2(6)-F1(4)*F2(5)-P3(1)*OM3*TMP2)
      V3(5)= DENOM*(-CI)*(-CI*(F1(3)*F2(6))+CI*(F1(4)*F2(5))-P3(2)*OM3
     $ *TMP2)
      V3(6)= DENOM*(-CI)*(-F1(3)*F2(5)-P3(3)*OM3*TMP2+F1(4)*F2(6))
      END


C     This File is Automatically generated by ALOHA 
C     The process calculated in this file is:
C     Gamma(3,2,-1)*ProjM(-1,1)
C
      SUBROUTINE FFV2_4_3(F1, F2, COUP1, COUP2, M3, W3,V3)
      IMPLICIT NONE
      COMPLEX*16 CI
      PARAMETER (CI=(0D0,1D0))
      COMPLEX*16 COUP1
      COMPLEX*16 COUP2
      COMPLEX*16 F1(*)
      COMPLEX*16 F2(*)
      REAL*8 M3
      REAL*8 OM3
      REAL*8 P3(0:3)
      COMPLEX*16 V3(6)
      COMPLEX*16 VTMP(6)
      REAL*8 W3
      COMPLEX*16 DENOM
      INTEGER*4 I
      CALL FFV2_3(F1,F2,COUP1,M3,W3,V3)
      CALL FFV4_3(F1,F2,COUP2,M3,W3,VTMP)
      DO I = 3, 6
        V3(I) = V3(I) + VTMP(I)
      ENDDO
      END
      

"""
        text = open(os.path.join(self.out_dir,'Source', 'DHELAS', 'FFV2_3.f')).read()
        self.assertIn('OM3', text)
        ffv2 = [l.strip() for l in ffv2.strip().split('\n')]
        text = [l.strip() for l in text.strip().split('\n')]
        self.assertEqual(ffv2, text) 
        
        
        
    def test_define_order(self):
        """Test the reordering of particles in the define"""

        self.do('import model sm')
        self.do('define p = u c~ g d s b~ b h')
        self.assertEqual(self.cmd._multiparticles['p'],
                         [21, 2, 1, 3, -4, 5, -5, 25])
        self.do('import model sm-no_masses')
        self.do('define p = u c~ g d s b~ b h')
        self.assertEqual(self.cmd._multiparticles['p'],
                         [21, 2, 1, 3, 5, -4, -5, 25])

    def test_madevent_decay_chain(self):
        """Test decay chain output"""

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        self.do('import model sm')
        self.do('define p = u u~ d d~')
        self.do('set group_subprocesses False')
        self.do('generate p p > w+, w+ > l+ vl @1')
        self.do('output madevent %s ' % self.out_dir)
        devnull = open(os.devnull,'w')
        # Check that all subprocess directories have been created
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P1_dxu_wp_wp_epve')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P1_udx_wp_wp_epve')))
        # Check that the Source directory compiles
        status = subprocess.call(['make'],
                                 stdout=devnull, 
                                 cwd=os.path.join(self.out_dir, 'Source'))

        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libdhelas.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libmodel.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libgeneric.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libcernlib.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libdsample.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libpdf.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libbias.a')))
        
        
        # Check that gensym compiles
        status = subprocess.call(['make', 'gensym'],
                                 stdout=devnull, 
                                 cwd=os.path.join(self.out_dir, 'SubProcesses',
                                                  'P1_udx_wp_wp_epve'))
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P1_udx_wp_wp_epve',
                                                    'gensym')))
        # Check that gensym runs
        proc = subprocess.Popen('./gensym',
                                  stdin=subprocess.PIPE, 
                                 stdout=devnull,
                                 cwd=os.path.join(self.out_dir, 'SubProcesses',
                                                  'P1_udx_wp_wp_epve'),
                                 shell=True)
        proc.communicate('100 4 0.1 .false.\n'.encode())
        
        self.assertEqual(proc.returncode, 0)
        # Check that madevent compiles
        status = subprocess.call(['make', 'madevent'],
                                 stdout=devnull, 
                                 cwd=os.path.join(self.out_dir, 'SubProcesses',
                                                  'P1_udx_wp_wp_epve'))
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P1_udx_wp_wp_epve',
                                                    'madevent')))
        
    def test_complex_mass_SA(self):
        """ Test that the complex_mass compile in fortran """

        self.do('import model sm --noprefix')
        self.do('set complex_mass_scheme')
        self.do('generate e+ e- > e+ e-')
        self.do('output standalone %s ' % self.out_dir)
        misc.compile(cwd=os.path.join(self.out_dir,'SubProcesses', 'P0_epem_epem'))
        p = subprocess.Popen(['./check'], cwd=os.path.join(self.out_dir,'SubProcesses', 'P0_epem_epem'),
                            stdout=subprocess.PIPE)
        #output = p.stdout.read()
        for line in p.stdout:
            line = line.decode('utf8')
            if 'Matrix element' in line:
                value = line.split('=')[1]
                value = value. split('GeV')[0]
                value = eval(value)
                self.assertAlmostEqual(value, 0.019538610404713896)
        
        self.do('import model sm')
        self.do('set complex_mass_scheme')
        self.do('generate e+ e- > e+ e-')
        self.do('output standalone %s -f' % self.out_dir)
        misc.compile(cwd=os.path.join(self.out_dir,'SubProcesses', 'P0_epem_epem'))
        p = subprocess.Popen(['./check'], cwd=os.path.join(self.out_dir,'SubProcesses', 'P0_epem_epem'),
                            stdout=subprocess.PIPE)
        #output = p.stdout.read()
        for line in p.stdout:
            line = line.decode('utf8')
            if 'Matrix element' in line:
                value = line.split('=')[1]
                value = value. split('GeV')[0]
                value = eval(value)
                self.assertAlmostEqual(value, 0.019538610404713896)

    def test_load_feynman(self):
        """ Test that feynman gauge assignment works """
        
        self.do('import model sm')
        # check that the model is correctly loaded (has some goldstone)
        nb_goldstone = 0
        for part in self.cmd._curr_model['particles']:
            if part.get('pdg_code') in [250, 251]:
                nb_goldstone += 1
        self.assertEqual(nb_goldstone, 0)
        self.do('set gauge Feynman')
        self.do('import model sm')
        # check that the model is correctly loaded (has some goldstone)
        nb_goldstone = 0
        for part in self.cmd._curr_model['particles']:
            if part.get('pdg_code') in [250, 251]:
                nb_goldstone += 1
        self.assertEqual(nb_goldstone, 2)
        

    def test_madevent_subproc_group(self):
        """Test MadEvent output using the SubProcess group functionality"""

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        self.do('import model sm')
        self.do('define p = g u d u~ d~')
        self.do('set group_subprocesses True')
        self.do('generate g g > p p @2')
        self.do('output madevent %s ' % self.out_dir)
        self.do('set group_subprocesses False')
        devnull = open(os.devnull,'w')
        # Check that all subprocess directories have been created
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P2_gg_gg')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P2_gg_qq')))
        if misc.which('gs'):
            self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P2_gg_qq',
                                                    'matrix11.jpg')))
            self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'HTML',
                                                    'card.jpg')))
        # Check that the run_config.inc file has been modified correctly
        run_config = open(os.path.join(self.out_dir, 'Source',
                                       'run_config.inc')).read()
        self.assertTrue(run_config.find("ChanPerJob=2"))
        generate_events = open(os.path.join(self.out_dir, 'bin',
                                       'generate_events')).read()
        self.assertTrue(generate_events.find(\
                                            "$dirbin/refine $a $mode $n 1 $t"))
        # Check that the maxconfigs.inc file has been created properly
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'Source',
                                                    'maxconfigs.inc')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P2_gg_qq',
                                                    'maxconfigs.inc')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P2_gg_qq',
                                                    'get_color.f')))
        # Check that the Source directory compiles
        status = subprocess.call(['make'],
                                 stdout=devnull, 
                                 cwd=os.path.join(self.out_dir, 'Source'))
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libdhelas.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libmodel.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libgeneric.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libcernlib.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libdsample.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libpdf.a')))
        # Check that combine_events, gen_ximprove, combine_runs 
        # compile
        #status = subprocess.call(['make', '../bin/internal/combine_events'],
        #                         stdout=devnull, 
        #                         cwd=os.path.join(self.out_dir, 'Source'))
        #self.assertEqual(status, 0)
        #self.assertTrue(os.path.exists(os.path.join(self.out_dir,
        #                                       'bin','internal', 'combine_events')))
        status = subprocess.call(['make', '../bin/internal/gen_ximprove'],
                                 stdout=devnull, 
                                 cwd=os.path.join(self.out_dir, 'Source'))
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'bin','internal', 'gen_ximprove')))
        # Check that gensym compiles
        status = subprocess.call(['make', 'gensym'],
                                 stdout=devnull, 
                                 cwd=os.path.join(self.out_dir, 'SubProcesses',
                                                  'P2_gg_qq'))
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P2_gg_qq',
                                                    'gensym')))
        # Check that gensym runs
        proc = subprocess.Popen('./gensym', 
                                 stdout=devnull, stdin=subprocess.PIPE,
                                 cwd=os.path.join(self.out_dir, 'SubProcesses',
                                                  'P2_gg_qq'), shell=True)
        proc.communicate('100 4 0.1 .false.\n'.encode())
        self.assertEqual(proc.returncode, 0)
        # Check that madevent compiles
        status = subprocess.call(['make', 'madevent_forhel'],
                                 stdout=devnull, 
                                 cwd=os.path.join(self.out_dir, 'SubProcesses',
                                                  'P2_gg_qq'))
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P2_gg_qq',
                                                    'madevent_forhel')))
        
    def test_madevent_subproc_group_symmetry(self):
        """Check that symmetry.f gives right output"""

        def analyse(fsock):
            data = []
            for line in fsock:
                if line.strip():
                    data.append([int(i) for i in line.split()])
            return data

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        self.do('import model MSSM_SLHA2')
        self.do('define q = u d u~ d~')
        self.do('set group_subprocesses True')
        self.do('generate u u~ > g > go go, go > q q n1 / ur dr')
        self.do('output %s ' % self.out_dir)
        self.do('set group_subprocesses False')
        devnull = open(os.devnull,'w')
        # Check that all subprocess directories have been created
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P0_qq_gogo_go_qqn1_go_qqn1')))
        
        target=""" 1   1
 2  -1
 3  -1
 4  -1
 5   1
 6  -5
 7  -5
 8  -5
 9   1
10  -9
11  -9
12  -9
"""
        
        self.assertEqual(analyse(target.split('\n')), 
                         analyse(open(os.path.join(self.out_dir,
                                           'SubProcesses',
                                           'P0_qq_gogo_go_qqn1_go_qqn1',
                                           'symfact_orig.dat'))))

        # Compile the Source directory
        status = subprocess.call(['make'],
                                 stdout=devnull, 
                                 cwd=os.path.join(self.out_dir, 'Source'))
        self.assertEqual(status, 0)

        # Compile gensym
        status = subprocess.call(['make', 'gensym'],
                                 stdout=devnull, 
                                 cwd=os.path.join(self.out_dir, 'SubProcesses',
                                                  'P0_qq_gogo_go_qqn1_go_qqn1'))
        # Run gensym
        proc = subprocess.Popen('./gensym', 
                                 stdout=devnull, stdin=subprocess.PIPE,
                                 cwd=os.path.join(self.out_dir, 'SubProcesses',
                                                  'P0_qq_gogo_go_qqn1_go_qqn1'), shell=True)
        proc.communicate('100 4 0.1 .false.\n'.encode())
        self.assertEqual(proc.returncode, 0)



        target ="""   1   1
   2  -1
   3  -1
   4  -1
   5   1
   6  -5
   7  -5
   8  -5
   9   1
  10  -9
  11  -9
  12  -9
"""
            
        # Check the new contents of the symfact.dat file
        self.assertEqual(analyse(open(os.path.join(self.out_dir,
                                           'SubProcesses',
                                           'P0_qq_gogo_go_qqn1_go_qqn1',
                                           'symfact.dat'))), 
                         analyse(target.split('\n')))

    def test_madevent_subproc_group_decay_chain(self):
        """Test decay chain output using the SubProcess group functionality"""

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        self.do('import model sm')
        self.do('define p = g u d u~ d~')
        self.do('set group_subprocesses True')
        self.do('generate p p > w+, w+ > l+ vl @1')
        self.do('add process p p > w+ p, w+ > l+ vl @2')
        self.do('output madevent %s -nojpeg' % self.out_dir)
        self.do('set group_subprocesses False')
        devnull = open(os.devnull,'w')
        # Check that all subprocess directories have been created
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P2_gq_wpq_wp_lvl')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P2_gq_wpq_wp_lvl')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P2_qq_wpg_wp_lvl')))
        goal_subproc_mg = \
"""P2_gq_wpq_wp_lvl
P2_qq_wpg_wp_lvl
P1_qq_wp_wp_lvl
"""
        self.assertFileContains(os.path.join(self.out_dir,
                                             'SubProcesses',
                                             'subproc.mg'),
                                goal_subproc_mg)
        # Check that the Source directory compiles
        status = subprocess.call(['make'],
                                 stdout=devnull, 
                                 cwd=os.path.join(self.out_dir, 'Source'))
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libdhelas.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libmodel.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libgeneric.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libcernlib.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libdsample.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libpdf.a')))
        # Check that gensym compiles
        status = subprocess.call(['make', 'gensym'],
                                 stdout=devnull, 
                                 cwd=os.path.join(self.out_dir, 'SubProcesses',
                                                  'P2_qq_wpg_wp_lvl'))
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P2_qq_wpg_wp_lvl',
                                                    'gensym')))
        # Check that gensym runs
        proc = subprocess.Popen('./gensym', 
                                 stdout=devnull, stdin=subprocess.PIPE,
                                 cwd=os.path.join(self.out_dir, 'SubProcesses',
                                                  'P2_qq_wpg_wp_lvl'),
                                 shell=True)
        proc.communicate('100 4 0.1 .false.\n'.encode())
        self.assertEqual(proc.returncode, 0)
        # Check that madevent compiles
        status = subprocess.call(['make', 'madevent_forhel'],
                                 stdout=devnull, 
                                 cwd=os.path.join(self.out_dir, 'SubProcesses',
                                                  'P2_qq_wpg_wp_lvl'))
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P2_qq_wpg_wp_lvl',
                                                    'madevent_forhel')))
        
    def test_ungroup_decay(self):
        """Test group_subprocesses=False for decay process"""

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        self.do('import model sm')
        self.do('set group_subprocesses False')
        self.do('generate w+ > l+ vl')
        self.do('add process w+ > j j')
        self.do('output %s ' % self.out_dir)
        # Check that all subprocesses have separate directories
        directories = ['P0_wp_epve','P0_wp_udx']
        for d in directories:
            self.assertTrue(os.path.isdir(os.path.join(self.out_dir,
                                                       'SubProcesses',
                                                       d)))
        self.do('set group_subprocesses True')
        self.do('generate w+ > l+ vl')
        self.do('add process w+ > j j')
        self.do('output %s -f' % self.out_dir)
        # Check that all subprocesses are combined
        directories = ['P0_wp_lvl','P0_wp_qq']
        for d in directories:
            self.assertTrue(os.path.isdir(os.path.join(self.out_dir,
                                                       'SubProcesses',
                                                       d)))
    
    @test_manager.bypass_for_py3
    def test_madevent_triplet_diquarks(self):
        """Test MadEvent output of triplet diquarks"""

        self.do('import model triplet_diquarks')
        self.do('set group_subprocesses False')
        self.do('generate u t > trip~ > u t g')
        self.do('output %s ' % self.out_dir)

        devnull = open(os.devnull,'w')
        # Check that the Source directory compiles
        status = subprocess.call(['make'],
                                 stdout=devnull, 
                                 cwd=os.path.join(self.out_dir, 'Source'))
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libdhelas.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libmodel.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libgeneric.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libcernlib.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libdsample.a')))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libpdf.a')))
        # Check that gensym compiles
        status = subprocess.call(['make', 'gensym'],
                                 stdout=devnull, 
                                 cwd=os.path.join(self.out_dir, 'SubProcesses',
                                                  'P0_ut_tripx_utg'))
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P0_ut_tripx_utg',
                                                    'gensym')))
        # Check that gensym runs
        proc = subprocess.Popen('./gensym', 
                                 stdout=devnull, stdin=subprocess.PIPE,
                                 cwd=os.path.join(self.out_dir, 'SubProcesses',
                                                  'P0_ut_tripx_utg'), shell=True)
        proc.communicate('100 4 0.1 .false.\n'.encode())
        self.assertEqual(proc.returncode, 0)
        
        # Check that madevent compiles
        status = subprocess.call(['make', 'madevent'],
                                 stdout=devnull, 
                                 cwd=os.path.join(self.out_dir, 'SubProcesses',
                                                  'P0_ut_tripx_utg'))
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P0_ut_tripx_utg',
                                                    'madevent')))
        
    def test_leshouche_sextet_diquarks(self):
        """Test leshouche.inc output of sextet diquarks"""

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        devnull = open(os.devnull,'w')

        # Test sextet production
        self.do('import model sextet_diquarks')
        self.do('set group_subprocesses False')
        self.do('generate u u > six g')
        self.do('output %s ' % self.out_dir)
        
        # Check that leshouche.inc exists
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P0_uu_sixg',
                                                    'leshouche.inc')))        
        # Test sextet decay
        self.do('generate six > u u g')
        self.do('output %s -f' % self.out_dir)

        # Check that leshouche.inc exists
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P0_six_uug',
                                                    'leshouche.inc')))        

        # Test sextet production
        self.do('generate u g > six u~')
        self.do('output %s -f' % self.out_dir)
        
        # Check that leshouche.inc exists
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P0_ug_sixux',
                                                    'leshouche.inc')))
    def test_ufo_standard_sm(self):
        """ check that we can use standard MG4 name """
        self.do('import model sm')
        self.do('generate mu+ mu- > ta+ ta-')       

    def test_decay_chain_identical_particle_outoforder(self):
        """ check that we can use standard MG4 name """
        
        self.do('import model sm')
        self.do('generate e+ e- > z z h, h > b b~, z > u u~, z > e+ e-')
        self.assertEqual(len(self.cmd._curr_amps), 1)
        self.do('output madevent %s ' % self.out_dir)
        Pdir = os.listdir(pjoin(self.out_dir, 'SubProcesses')) 
        self.assertNotIn('P0_ll_zzh_z_ll_z_ll_h_bbx',  Pdir)





    def test_save_load(self):
        """ check that we can use standard MG4 name """
        
        self.do('import model sm')
        self.assertEqual(len(self.cmd._curr_model.get('particles')), 17)
        self.assertEqual(len(self.cmd._curr_model.get('interactions')), 56)
        self.do('save model /tmp/model.pkl')
        self.do('import model MSSM_SLHA2-full')
        self.do('load model /tmp/model.pkl')
        self.assertEqual(len(self.cmd._curr_model.get('particles')), 17)
        self.assertEqual(len(self.cmd._curr_model.get('interactions')), 56)
        self.do('generate mu+ mu- > ta+ ta-') 
        self.assertEqual(len(self.cmd._curr_amps), 1)
        nicestring = """Process: mu+ mu- > ta+ ta- WEIGHTED<=4
2 diagrams:
1  ((1(13),2(-13)>1(22),id:35),(3(-15),4(15),1(22),id:36)) (QCD=0,QED=2,WEIGHTED=4)
2  ((1(13),2(-13)>1(23),id:41),(3(-15),4(15),1(23),id:42)) (QCD=0,QED=2,WEIGHTED=4)"""

        self.assertEqual(self.cmd._curr_amps[0].nice_string().split('\n'), nicestring.split('\n'))
        self.do('save processes /tmp/model.pkl')
        self.do('generate e+ e- > e+ e-')
        self.do('load processes /tmp/model.pkl')
        self.assertEqual(len(self.cmd._curr_amps), 1)
        self.assertEqual(self.cmd._curr_amps[0].nice_string(), nicestring)
        
        os.remove('/tmp/model.pkl')
        
    def test_pythia8_output(self):
        """Test Pythia 8 output"""

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)
        # Create out_dir and out_dir/include
        os.makedirs(os.path.join(self.out_dir,'include'))
        # Touch the file Pythia.h, which is needed to verify that this is a Pythia dir
        py_h_file = open(os.path.join(self.out_dir,'include','Pythia.h'), 'w')
        py_h_file.close()

        self.do('import model sm')
        self.do('define p g u d u~ d~')
        self.do('define j g u d u~ d~')
        self.do('generate p p > w+ j @2')
        self.do('output pythia8 %s' % self.out_dir)
        # Check that the needed files are generated
        files = ['Processes_sm/Sigma_sm_gq_wpq.h', 'Processes_sm/Sigma_sm_gq_wpq.cc',
                 'Processes_sm/Sigma_sm_qq_wpg.h', 'Processes_sm/Sigma_sm_qq_wpg.cc',
                 'Processes_sm/HelAmps_sm.h', 'Processes_sm/HelAmps_sm.cc',
                 'Processes_sm/Parameters_sm.h',
                 'Processes_sm/Parameters_sm.cc', 'Processes_sm/Makefile',
                 'examples/main_sm_1.cc', 'examples/Makefile_sm_1']
        for f in files:
            self.assertTrue(os.path.isfile(os.path.join(self.out_dir, f)), 
                            '%s file is not in directory' % f)
        self.do('generate u u~ > a a a a')
        self.assertRaises(MadGraph5Error,
                          self.do,
                          'output pythia8 %s' % self.out_dir)
        self.do('generate u u~ > w+ w-, w+ > e+ ve, w- > e- ve~ @1')
        self.assertRaises(MadGraph5Error,
                          self.do,
                          'output pythia8 %s' % self.out_dir)

    def test_standalone_cpp_output(self):
        """Test the C++ standalone output"""

        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)

        self.do('import model sm')
        self.do('generate e+ e- > e+ e- @2')
        self.do('output standalone_cpp %s' % self.out_dir)

        # Check that all needed src files are generated
        files = ['HelAmps_sm.h', 'HelAmps_sm.cc', 'Makefile',
                 'Parameters_sm.h', 'Parameters_sm.cc',
                 'rambo.h', 'rambo.cc', 'read_slha.h', 'read_slha.cc']

        for f in files:
            self.assertTrue(os.path.isfile(os.path.join(self.out_dir,
                                                       'src',
                                                        f)), 
                            '%s file is not in aloha directory' % f)

        devnull = open(os.devnull,'w')
        # Check that the Model and Aloha output has compiled
        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                               'lib', 'libmodel_sm.a')))
        # Check that check_sa.cpp compiles
        subprocess.call(['make', 'check'],
                        stdout=devnull, stderr=devnull, 
                        cwd=os.path.join(self.out_dir, 'SubProcesses',
                                         'P2_Sigma_sm_epem_epem'))


        self.assertTrue(os.path.exists(os.path.join(self.out_dir,
                                                    'SubProcesses',
                                                    'P2_Sigma_sm_epem_epem',
                                                    'check')))

        # Check that the output of check is correct 
        logfile = os.path.join(self.out_dir, 'SubProcesses',
                               'P2_Sigma_sm_epem_epem', 'check.log')

        subprocess.call('./check', 
                        stdout=open(logfile, 'w'), stderr=devnull,
                        cwd=os.path.join(self.out_dir, 'SubProcesses',
                                         'P2_Sigma_sm_epem_epem'), shell=True)

        log_output = open(logfile, 'r').read()
        me_re = re.compile(r'Matrix element\s*=\s*(?P<value>[\d\.e\+-]+)\s*GeV',
                           re.IGNORECASE)
        me_groups = me_re.search(log_output)
        self.assertTrue(me_groups)
        self.assertAlmostEqual(float(me_groups.group('value')), 0.019455844550069087)
        
    def test_import_banner_command(self):
        """check that the import banner command works"""
        
        cwd = os.getcwd()
        os.chdir(self.tmpdir)
        self.do('import banner %s --no_launch' % pjoin(MG5DIR, 'tests', 'input_files', 'tt_banner.txt'))
        
        # check that the output exists:
        self.assertTrue(os.path.exists(self.out_dir))
        
        # check that the Cards have been modified
        run_card = open(pjoin(self.out_dir,'Cards','run_card.dat')).read()
        self.assertIn("'tt'     = run_tag", run_card)
        self.assertIn("200       = nevents", run_card)
        os.chdir(cwd)
        
