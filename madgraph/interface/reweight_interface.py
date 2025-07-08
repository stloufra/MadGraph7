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
""" Command interface for Re-Weighting """
from __future__ import division
from __future__ import absolute_import
import difflib
import logging
import math
import os
import re
import shutil
import sys
import tempfile
import time
import subprocess
from subprocess import Popen, PIPE, STDOUT
from six.moves import map
from six.moves import range
from six.moves import zip
import six


pjoin = os.path.join

import madgraph
import madgraph.interface.extended_cmd as extended_cmd
import madgraph.interface.madgraph_interface as mg_interface
import madgraph.interface.master_interface as master_interface
import madgraph.interface.common_run_interface as common_run_interface
import madgraph.interface.madevent_interface as madevent_interface
import madgraph.iolibs.files as files
#import MadSpin.interface_madspin as madspin_interface
import madgraph.various.misc as misc
import madgraph.various.banner as banner
import madgraph.various.lhe_parser as lhe_parser
import madgraph.various.combine_plots as combine_plots
import madgraph.various.cluster as cluster
import madgraph.fks.fks_common as fks_common
import madgraph.core.diagram_generation as diagram_generation

import models.import_ufo as import_ufo
import models.check_param_card as check_param_card 
#import MadSpin.decay as madspin


logger = logging.getLogger('decay.stdout') # -> stdout
logger_stderr = logging.getLogger('decay.stderr') # ->stderr
cmd_logger = logging.getLogger('cmdprint2') # -> print

# global to check which f2py module have been already loaded. (to avoid border effect)
dir_to_f2py_free_mod = {}
nb_f2py_module = 0 # each time the process/model is changed this number is modified to 
                   # forced the python module to re-create an executable

#lhapdf = None


class ReweightInterface(extended_cmd.Cmd):
    """Basic interface for reweighting operation"""
    
    prompt = 'Reweight>'
    debug_output = 'Reweight_debug'
    
    @misc.mute_logger()
    def __init__(self, event_path=None, allow_madspin=False, mother=None, *completekey, **stdin):
        """initialize the interface with potentially an event_path"""
        
        
        self.me_dir = os.getcwd()
        if not event_path:
            cmd_logger.info('************************************************************')
            cmd_logger.info('*                                                          *')
            cmd_logger.info('*               Welcome to Reweight Module                 *')
            cmd_logger.info('*                                                          *')
            cmd_logger.info('************************************************************')
        extended_cmd.Cmd.__init__(self, *completekey, **stdin)
        
        self.model = None
        self.has_standalone_dir = False
        self.mother= mother # calling interface
        self.multicore=False
        
        self.options = {'curr_dir': os.path.realpath(os.getcwd()),
                        'rwgt_name':None,
                        "allow_missing_finalstate":False,
                        "identical_particle_in_prod_and_decay": "average"}

        self.events_file = None
        self.processes = {}
        self.f2pylib = {}
        self.second_model = None
        self.second_process = None
        self.nb_library = 1
        self.dedicated_path = {}
        self.soft_threshold = None
        self.systematics = False # allow to run systematics in ouput2.0 mode
        self.boost_event = False
        self.mg5cmd = master_interface.MasterCmd()
        if mother:
            self.mg5cmd.options.update(mother.options)
        self.seed = None
        self.output_type = "default"
        self.helicity_reweighting = True
        self.rwgt_mode = '' # can be LO, NLO, NLO_tree, '' is default 
        self.has_nlo = False
        self.rwgt_dir = None
        self.exitted = False # Flag to know if do_quit was already called.
        self.keep_ordering = False
        self.use_eventid = False
        if event_path:
            logger.info("Extracting the banner ...")
            self.do_import(event_path, allow_madspin=allow_madspin)
            
        # dictionary to fortan evaluator
        self.calculator = {}
        self.calculator_nbcall = {}
        
        #all the cross-section for convenience
        self.all_cross_section = {}

        #If we are using the DensityInterface
        self.flag_density_matrix = False
            
    def do_import(self, inputfile, allow_madspin=False):
        """import the event file"""

        args = self.split_arg(inputfile)
        if not args:
            return self.InvalidCmd, 'import requires arguments'
        
        # change directory where to write the output
        self.options['curr_dir'] = os.path.realpath(os.path.dirname(inputfile))
        if os.path.basename(os.path.dirname(os.path.dirname(inputfile))) == 'Events':
            self.options['curr_dir'] = pjoin(self.options['curr_dir'], 
                                                      os.path.pardir, os.pardir)
            
        
        if not os.path.exists(inputfile):
            if inputfile.endswith('.gz'):
                if not os.path.exists(inputfile[:-3]):
                    raise self.InvalidCmd('No such file or directory : %s' % inputfile)
                else: 
                    inputfile = inputfile[:-3]
            elif os.path.exists(inputfile + '.gz'):
                inputfile = inputfile + '.gz'
            else: 
                raise self.InvalidCmd('No such file or directory : %s' % inputfile)
        
        if inputfile.endswith('.gz'):
            misc.gunzip(inputfile)
            inputfile = inputfile[:-3]

        # Read the banner of the inputfile
        self.lhe_input = lhe_parser.EventFile(os.path.realpath(inputfile))
        if not self.lhe_input.banner:
            value = self.ask("What is the path to banner", 0, [0], "please enter a path", timeout=0)
            self.lhe_input.banner = open(value).read()
        self.banner = self.lhe_input.get_banner()
        
        #get original cross-section/error
        if 'init' not in self.banner:
            self.orig_cross = (0,0)
            #raise self.InvalidCmd('Event file does not contain init information')
        else:
            for line in self.banner['init'].split('\n'):
                    split = line.split()
                    if len(split) == 4:
                        cross, error = float(split[0]), float(split[1])
            self.orig_cross = (cross, error)
        
        
        
        # Check the validity of the banner:
        if 'slha' not in self.banner:
            self.events_file = None
            raise self.InvalidCmd('Event file does not contain model information')
        elif 'mg5proccard' not in self.banner:
            self.events_file = None
            raise self.InvalidCmd('Event file does not contain generation information')

        if 'madspin' in self.banner and not allow_madspin:
            raise self.InvalidCmd('Reweight should be done before running MadSpin')
        
                
        # load information
        process = self.banner.get_detail('proc_card', 'generate')
        if '[' in process and isinstance(self.banner.get('run_card'), banner.RunCardNLO):
            if not self.banner.get_detail('run_card', 'store_rwgt_info'):
                logger.warning("The information to perform a proper NLO reweighting is not present in the event file.")
                logger.warning("       We will perform a LO reweighting instead. This does not guarantee NLO precision.")
                self.rwgt_mode = 'LO'

            if self.mother and 'OLP' in self.mother.options:
                if self.mother.options['OLP'].lower() != 'madloop':
                    logger.warning("Accurate NLO mode only works for OLP=MadLoop not for OLP=%s. An approximate (LO) reweighting will be performed instead")
                    self.rwgt_mode = 'LO'
            
            if self.mother and 'lhapdf' in self.mother.options and not self.mother.options['lhapdf']:
                logger.warning('NLO accurate reweighting requires lhapdf to be installed. Pass in approximate LO mode.')
                self.rwgt_mode = 'LO'
        else:
            self.rwgt_mode = 'LO'

        if not process:
            msg = 'Invalid proc_card information in the file (no generate line):\n %s' % self.banner['mg5proccard']
            raise Exception(msg)
        process, option = mg_interface.MadGraphCmd.split_process_line(process)
        self.proc_option = option
        self.is_decay = len(process.split('>',1)[0].split()) == 1 
        
        logger.info("process: %s" % process)
        logger.info("options: %s" % option)

    @staticmethod
    def get_LO_definition_from_NLO(proc, model, real_only=False):
        """return the LO definitions of the process corresponding to the born/real"""
        
        # split the line definition with the part before and after the NLO tag
        process, order, final = re.split(r'\[\s*(.*)\s*\]', proc)
        if process.strip().startswith(('generate', 'add process')):
            process = process.replace('generate', '')
            process = process.replace('add process','')
        
        # add the part without any additional jet.
        commandline="add process %s %s --no_warning=duplicate;" % (process, final)
        if not order:
            #NO NLO tag => nothing to do actually return input
            return proc
        elif not order.startswith(('virt','LOonly','noborn')):
            # OK this a standard NLO process            
            if real_only:
                commandline= '' 
            
            if '=' in order:
                # get the type NLO QCD/QED/...
                order = order.split('=',1)[1].strip()

            # define the list of particles that are needed for the radiation
            pert = fks_common.find_pert_particles_interactions(model,
                                           pert_order = order)['soft_particles']
            commandline += "define pert_%s = %s;" % (order.replace(' ',''), ' '.join(map(str,pert)) )
            
            # check if we have to increase by one the born order
            
            if '%s=' % order in process or '%s<=' % order in process:
                result=re.split(' ',process)
                process=''
                for r in result:
                    if '%s=' % order in r:
                        ior=re.split('=',r)
                        r='QCD=%i' % (int(ior[1])+1)
                    elif '%s<=' % order in r:
                        ior=re.split('=',r)
                        r='QCD<=%i' % (int(ior[1])+1)
                    process=process+r+' '
            #handle special tag $ | / @
            result = re.split(r'([/$@]|\w+(?:^2)?(?:=|<=|>)+\w+)', process, 1)                    
            if len(result) ==3:
                process, split, rest = result
                commandline+="add process %s pert_%s %s%s %s --no_warning=duplicate;" % (process, order.replace(' ','') ,split, rest, final)
            else:
                commandline +='add process %s pert_%s %s --no_warning=duplicate;' % (process,order.replace(' ',''), final)
        elif order.startswith(('noborn')):
            # pass in sqrvirt=
            return "add process %s [%s] %s;" % (process, order.replace('noborn', 'sqrvirt'), final)
        elif order.startswith('LOonly'):
            #remove [LOonly] flag
            return "add process %s %s;" % (process, final)
        else:
            #just return the input. since this Madloop.
            if order:
                return "add process %s [%s] %s ;" % (process, order,final)
            else:
                return "add process %s %s ;" % (process, final)
        return commandline


    def check_events(self):
        """Check some basic property of the events file"""
        
        sum_of_weight = 0
        sum_of_abs_weight = 0
        negative_event = 0
        positive_event = 0
        
        start = time.time()
        for event_nb,event in enumerate(self.lhe_input):
            #control logger
            if (event_nb % max(int(10**int(math.log10(float(event_nb)+1))),10)==0): 
                    running_time = misc.format_timer(time.time()-start)
                    logger.info('Event nb %s %s' % (event_nb, running_time))
            if (event_nb==10001): logger.info('reducing number of print status. Next status update in 10000 events')

            try:
                event.check() #check 4 momenta/...
            except Exception as error:
                print(event)
                raise error
            sum_of_weight += event.wgt
            sum_of_abs_weight += abs(event.wgt)
            if event.wgt < 0 :
                negative_event +=1
            else:
                positive_event +=1
        
        logger.info("total cross-section: %s" % sum_of_weight)
        logger.info("total abs cross-section: %s" % sum_of_abs_weight) 
        logger.info("fraction of negative event %s", negative_event/(negative_event+positive_event))      
        logger.info("total number of events %s", (negative_event+positive_event))
        logger.info("negative event %s", negative_event)
        
        
        
        
    @extended_cmd.debug()
    def complete_import(self, text, line, begidx, endidx):
        "Complete the import command"
        
        args=self.split_arg(line[0:begidx])
        
        if len(args) == 1:
            base_dir = '.'
        else:
            base_dir = args[1]
        
        return self.path_completion(text, base_dir)
        
        # Directory continuation
        if os.path.sep in args[-1] + text:
            return self.path_completion(text,
                                    pjoin(*[a for a in args if \
                                                      a.endswith(os.path.sep)]))
    
    def help_change(self):
        """help for change command"""
    
        print("change model X :use model X for the reweighting")
        print("change process p p > e+ e-: use a new process for the reweighting")
        print("change process p p > mu+ mu- --add : add one new process to existing ones")
        print("change output [default|2.0|unweight]:")
        print("               default: add weight(s) to the current file")    
    
    def do_change(self, line):
        """allow to define a second model/processes"""
        
        global nb_f2py_module
        
        args = self.split_arg(line)
        if len(args)<2:
            logger.critical("not enough argument (need at least two). Discard line")
        if args[0] == "model":
            nb_f2py_module += 1 # tag to force the f2py to reload
            self.second_model = " ".join(args[1:])
            if self.has_standalone_dir:
                self.terminate_fortran_executables()
                self.has_standalone_dir = False
        elif args[0] in ["keep_ordering", "use_eventid"]:
            setattr(self, args[0], banner.ConfigFile.format_variable(args[1], bool, args[0]))
        elif args[0] == "allow_missing_finalstate":
            self.options["allow_missing_finalstate"] = banner.ConfigFile.format_variable(args[1], bool, "allow_missing_finalstate")
        elif args[0] == "process":
            nb_f2py_module += 1
            if self.has_standalone_dir:
                self.terminate_fortran_executables()
                self.has_standalone_dir = False
            if args[-1] == "--add":
                self.second_process.append(" ".join(args[1:-1]))
            else:
                self.second_process = [" ".join(args[1:])]
        elif args[0] == "boost":
            self.boost_event = eval(' '.join(args[1:]))
        elif args[0] in ['virtual_path', 'tree_path']:
            self.dedicated_path[args[0]] = os.path.abspath(args[1])
        elif args[0] == "output":
            if args[1] in ['default', '2.0', 'unweight']:
                self.output_type = args[1]
        elif args[0] == "helicity":
            self.helicity_reweighting = banner.ConfigFile.format_variable(args[1], bool, "helicity")
        elif args[0] == "mode":
            if args[1] != 'LO':
                if 'OLP' in self.mother.options and self.mother.options['OLP'].lower() != 'madloop':
                    logger.warning("Only LO reweighting is allowed for OLP!=MadLoop. Keeping the mode to LO.")
                    self.rwgt_mode = 'LO'
                elif not self.banner.get_detail('run_card','store_rwgt_info', default=False):
                    logger.warning("Missing information for NLO type of reweighting. Keeping the mode to LO.")
                    self.rwgt_mode = 'LO'
                elif 'lhapdf' in self.mother.options and not self.mother.options['lhapdf']:
                    logger.warning('NLO accurate reweighting requires lhapdf to be installed. Pass in approximate LO mode.')
                    self.rwgt_mode = 'LO'
                else:
                    self.rwgt_mode = args[1]
            else:
                self.rwgt_mode = args[1]
        elif args[0] == "rwgt_dir":
            self.rwgt_dir = args[1]
            if not os.path.exists(self.rwgt_dir):
                os.mkdir(self.rwgt_dir)
            self.rwgt_dir = os.path.abspath(self.rwgt_dir)
        elif args[0] == 'systematics':
            if self.output_type == 'default' and args[1].lower() not in ['none', 'off']:
                logger.warning('systematics can only be computed for non default output type. pass to output mode \'2.0\'')
                self.output_type = '2.0'
            if len(args) == 2:
                try:
                    self.systematics = banner.ConfigFile.format_variable(args[1], bool)
                except Exception as error:
                    self.systematics = args[1:]
            else:
                self.systematics = args[1:]
        elif args[0] == 'soft_threshold':
            self.soft_threshold = banner.ConfigFile.format_variable(args[1], float, 'soft_threshold')
        elif args[0] == 'multicore':
            pass 
            # this line is meant to be parsed by common_run_interface and change the way this class is called.
            #It has no direct impact on this class.
        elif args[0] == "identical_particle_in_prod_and_decay":
            if args[1].lower() not in ['average', 'max', 'crash']:
                raise Exception("option identical_particle_in_prod_and_decay can only be one of the following ['average', 'max', 'crash']")
            self.options[args[0]] = args[1].lower()
        else:
            logger.critical("unknown option! %s.  Discard line." % args[0])
        
             
    def check_launch(self, args):
        """check the validity of the launch command"""
        
        if not self.lhe_input:
            if isinstance(self.lhe_input, lhe_parser.EventFile):
                self.lhe_input = lhe_parser.EventFile(self.lhe_input.name)
            else:
                raise self.InvalidCmd("No events files defined.")
            
        opts = {'rwgt_name':None, 'rwgt_info':None}
        if any(a.startswith('--') for a in args):
            for a in args[:]:
                if a.startswith('--') and '=' in a:
                    key,value = a[2:].split('=')
                    opts[key] = value .replace("'","") .replace('"','')

        return opts

    def help_launch(self):
        """help for the launch command"""
        
        logger.info('''Add to the loaded events a weight associated to a 
        new param_card (to be define). The weight returned is the ratio of the 
        square matrix element by the squared matrix element of production.
        All scale are kept fix for this re-weighting.''')


    def get_weight_names(self):
        """ return the various name for the computed weights """
        
        if self.rwgt_mode == 'LO':
            return ['']
        elif self.rwgt_mode == 'NLO':
            return ['_nlo']
        elif self.rwgt_mode == 'LO+NLO':
            return ['_lo', '_nlo']
        elif self.rwgt_mode == 'NLO_tree':
            return ['_tree']        
        elif not self.rwgt_mode and self.has_nlo :
            return ['_nlo']
        else:
            return ['']

    @misc.mute_logger()
    def do_launch(self, line):
        """end of the configuration launched the code"""
        #misc.sprint(self.flag_density_matrix)
        args = self.split_arg(line)
        opts = self.check_launch(args)
        if opts['rwgt_name']:
            self.options['rwgt_name'] = opts['rwgt_name']
        if opts['rwgt_info']:
            self.options['rwgt_info'] = opts['rwgt_info']
        model_line = self.banner.get('proc_card', 'full_model_line')


        if not self.has_standalone_dir:                           
            if self.rwgt_dir and os.path.exists(pjoin(self.rwgt_dir,'rw_me','rwgt.pkl')):
                self.load_from_pickle()
                if opts['rwgt_name']:
                    self.options['rwgt_name'] = opts['rwgt_name']
                if not self.rwgt_dir:
                    self.me_dir = self.rwgt_dir
                self.load_module()       # load the fortran information from the f2py module
            elif self.multicore == 'wait':
                i=0
                while not os.path.exists(pjoin(self.me_dir,'rw_me','rwgt.pkl')):
                    time.sleep(10+i)
                    i+=5
                if not self.rwgt_dir:
                    self.rwgt_dir = self.me_dir
                self.load_from_pickle(keep_name=True)
                self.load_module()
            else:
                self.create_standalone_directory() 
                self.compile()
                self.load_module()  
                if self.multicore == 'create':
                    self.load_module()
                    if not self.rwgt_dir:
                        self.rwgt_dir = self.me_dir
                    self.save_to_pickle()      
        
        # get the mode of reweighting #LO/NLO/NLO_tree/...
        type_rwgt = self.get_weight_names() #type_rwgt = '' in my case
        # get iterator over param_card and the name associated to the current reweighting.
        param_card_iterator, tag_name = self.handle_param_card(model_line, args, type_rwgt)
        if self.rwgt_dir:
            path_me =self.rwgt_dir
        else:
            path_me = self.me_dir 
            
        if self.second_model or self.second_process or self.dedicated_path:
            rw_dir = pjoin(path_me, 'rw_me_%s' % self.nb_library)
        else:
            rw_dir = pjoin(path_me, 'rw_me')
        
        start = time.time()
        # initialize the collector for the various re-weighting
        cross, ratio, ratio_square,error = {},{},{}, {}
        for name in type_rwgt + ['orig']:
            cross[name], error[name] = 0.,0.
            ratio[name],ratio_square[name] = 0., 0.# to compute the variance and associate error
        
        if self.output_type == "default":
            if self.flag_density_matrix: #if density mode, we add a tag to the banner object containing density information and we delete the tag 'initrwgt'
                self.banner['MGDensity'] = 'helicity_direction = ' + str(self.helicity_direction) + '\n' + \
                                           'particle_in_density_matrix = ' + str(self.particle_in_density_matrix) + '\n' + \
                                           'momenta_boost = ' + str(self.momenta_boost) + '\n' + \
                                           'allowed_helicities = ' + str(self.allowed_helicities) + '\n' + \
                                           'number_changing_helicities = ' + str(self.number_changing_helicities) + '\n' + \
                                           'number_combinations = ' + str(self.number_combinations) + '\n' + \
                                           'axis_referential = ' + str(self.axis_referential) + '\n' + \
                                           'symmetrise_initial_state' + str(self.symmetrise_initial_state)
                self.banner.pop('initrwgt')
            output = open( self.lhe_input.path +'rw', 'w')
            #write the banner to the output file
            self.banner.write(output, close_tag=False)
            
        else:
            output = {}
            if tag_name.isdigit():
                name_tag= 'rwgt_%s' % tag_name
            else:
                name_tag = tag_name
            base = os.path.dirname(self.lhe_input.name)
            for rwgttype in  type_rwgt:
                output[(name_tag,rwgttype)] = lhe_parser.EventFile(pjoin(base,'rwgt_events%s_%s.lhe.gz' %(rwgttype,tag_name)), 'w')
                #write the banner to the output file
                self.banner.write(output[(name_tag,rwgttype)], close_tag=False)
                
        if self.lhe_input.closed:
            self.lhe_input = lhe_parser.EventFile(self.lhe_input.name)

        self.lhe_input.seek(0)
        for event_nb,event in enumerate(self.lhe_input):
            #control logger
            if (event_nb % max(int(10**int(math.log10(float(event_nb)+1))),10)==0): 
                    running_time = misc.format_timer(time.time()-start)
                    logger.info('Event nb %s %s' % (event_nb, running_time))
            if (event_nb==10001): logger.info('reducing number of print status. Next status update in 10000 events')
            if (event_nb==100001): logger.info('reducing number of print status. Next status update in 100000 events')
                
            weight = self.calculate_weight(event) #calculates the cross section or the production matrix

            #weight = {'orig': production matrix}
            if self.flag_density_matrix:
                import madgraph.various.Density_functions as dens
                density_matrix_to_print = dens.get_rho_normalised(weight['orig'], self.number_combinations).tolist()
                for elem in range(len(density_matrix_to_print)):
                    if density_matrix_to_print[elem].imag == 0:
                        density_matrix_to_print[elem] = float(density_matrix_to_print[elem].real)
                event.density = density_matrix_to_print #add the density information to the event for lhe_parser



            if not isinstance(weight, dict):
                weight = {'':weight}

            if self.flag_density_matrix:
                self.production_matrix += weight['orig'] * event.wgt

            for name in weight:
                cross[name] += weight[name]
                ratio[name] += weight[name]/event.wgt
                ratio_square[name] += (weight[name]/event.wgt)**2

            # ensure to have a consistent order of the weights. new one are put 
            # at the back, remove old position if already defines
            for tag in type_rwgt:
                try:
                    event.reweight_order.remove('%s%s'  % (tag_name,tag))
                except ValueError:
                    continue
            
            event.reweight_order += ['%s%s' % (tag_name,name) for name in type_rwgt]  
            if self.output_type == "default":
                for name in weight:
                    if 'orig' in name:
                        continue             
                    event.reweight_data['%s%s' % (tag_name,name)] = weight[name]
                    #write this event with weight
                # misc.sprint(event)
                output.write(str(event))
            else:
                for i,name in enumerate(weight):
                    if 'orig' in name:
                        continue 
                    if weight[name] == 0:
                        continue
                    new_evt = lhe_parser.Event(str(event))
                    new_evt.wgt = weight[name]
                    new_evt.parse_reweight()
                    new_evt.reweight_data = {}  
                    output[(tag_name,name)].write(str(new_evt))

        # check normalisation of the events:
        if self.run_card and 'event_norm' in self.run_card:
            if self.run_card['event_norm'] in ['average','bias']:
                for key, value in cross.items():
                    cross[key] = value / (event_nb+1)
                
        running_time = misc.format_timer(time.time()-start)
        logger.info('All event done  (nb_event: %s) %s' % (event_nb+1, running_time))        
        
        
        if self.output_type == "default":
            output.write('</LesHouchesEvents>\n')
            output.close()
        else:
            for key in output:
                output[key].write('</LesHouchesEvents>\n')
                output[key].close()
                if self.systematics and len(output) ==1:
                    try:
                        logger.info('running systematics computation')
                        import madgraph.various.systematics as syst
                        
                        if not isinstance(self.systematics, bool):
                            args = [output[key].name, output[key].name] + self.systematics
                        else:
                            args = [output[key].name, output[key].name]
                        if self.mother and self.mother.options['lhapdf']:
                            args.append('--lhapdf_config=%s' % self.mother.options['lhapdf'])
                        syst.call_systematics(args, result=open('rwg_syst_%s.result' % key[0],'w'),
                                              log=logger.info)
                    except Exception:
                        logger.error('fail to add systematics')
                        raise
##################################################################################################################
        # add output information        
        if self.mother and hasattr(self.mother, 'results'):
            run_name = self.mother.run_name
            results = self.mother.results
            results.add_run(run_name, self.run_card, current=True)
            results.add_detail('nb_event', event_nb+1)
            name = type_rwgt[0]
            results.add_detail('cross', cross[name])
            event_nb +=1
            for name in type_rwgt:
                variance = ratio_square[name]/event_nb - (ratio[name]/event_nb)**2
                orig_cross, orig_error = self.orig_cross
                error[name] = math.sqrt(max(0,variance/math.sqrt(event_nb))) * orig_cross + ratio[name]/event_nb * orig_error
            results.add_detail('error', error[type_rwgt[0]])
            import madgraph.interface.madevent_interface as ME_interface

        self.lhe_input.close()
        if not self.mother:
            name, ext = self.lhe_input.name.rsplit('.',1)
            target = '%s_out.%s' % (name, ext)            
        elif self.output_type != "default" :
            target = pjoin(self.mother.me_dir, 'Events', run_name, 'events.lhe')
        else:
            target = self.lhe_input.name
        
        if self.output_type == "default":
            files.mv(output.name, target)
            logger.info('Event %s have now the additional weight' % self.lhe_input.name)
        elif self.output_type == "unweight":
            for key in output:
                #output[key].write('</LesHouchesEvents>\n')
                #output.close()
                lhe = lhe_parser.EventFile(output[key].name)
                nb_event = lhe.unweight(target)
                if self.mother and  hasattr(self.mother, 'results'):
                    results = self.mother.results
                    results.add_detail('nb_event', nb_event)
                    results.current.parton.append('lhe')
                logger.info('Event %s is now unweighted under the new theory: %s(%s)' % (lhe.name, target, nb_event))                
        else:
            if self.mother and  hasattr(self.mother, 'results'):
                results = self.mother.results
                results.current.parton.append('lhe')       
            logger.info('Eventfiles is/are now created with new central weight')
        
        if self.multicore != 'create':
            for name in cross:
                if name == 'orig':
                    continue
                if not self.flag_density_matrix:
                    logger.info('new cross-section is %s: %g pb (indicative error: %g pb)' %\
                            ('(%s)' %name if name else '',cross[name], error[name]))
            
        self.terminate_fortran_executables(new_card_only=True)
        #store result
        for name in cross:
            if name == 'orig':
                self.all_cross_section[name] = (cross[name], error[name])
            else:
                self.all_cross_section[(tag_name,name)] = (cross[name], error[name])

        # perform the scanning
        if param_card_iterator:
            if self.options['rwgt_name']:
                reweight_name = self.options['rwgt_name'].rsplit('_',1)[0] # to avoid side effect during the scan
            else:
                reweight_name = None
            for i,card in enumerate(param_card_iterator):
                if reweight_name:
                    self.options['rwgt_name'] = '%s_%s' % (reweight_name, i+1)
                self.new_param_card = card
                #card.write(pjoin(rw_dir, 'Cards', 'param_card.dat'))
                self.exec_cmd("launch --keep_card", printcmd=False, precmd=True)
        
        self.options['rwgt_name'] = None


    def handle_param_card(self, model_line, args, type_rwgt):
        

        if self.rwgt_dir:
            path_me =self.rwgt_dir
        else:
            path_me = self.me_dir 
            
        if self.second_model or self.second_process or self.dedicated_path:
            rw_dir = pjoin(path_me, 'rw_me_%s' % self.nb_library)
        else:
            rw_dir = pjoin(path_me, 'rw_me')
        
        if not '--keep_card' in args:
            if self.has_nlo and self.rwgt_mode != "LO":
                rwdir_virt = rw_dir.replace('rw_me', 'rw_mevirt')
            with open(pjoin(rw_dir, 'Cards', 'param_card.dat'), 'w') as fsock:
                fsock.write(self.banner['slha']) 
            out, cmd = common_run_interface.CommonRunCmd.ask_edit_card_static(cards=['param_card.dat'], #Cette commande appelle les cartes de nouveau
                                   ask=self.ask, pwd=rw_dir, first_cmd=self.stored_line,
                                   write_file=False, return_instance=True
                                   )
            self.stored_line = None
            card = cmd.param_card
            new_card = card.write()
        elif self.new_param_card:
            new_card = self.new_param_card.write()
        else:
            new_card = open(pjoin(rw_dir, 'Cards', 'param_card.dat')).read()
        
        # check for potential scan in the new card 
        pattern_scan = re.compile(r'''^(decay)?[\s\d]*scan''', re.I+re.M) 
        param_card_iterator = []
        if pattern_scan.search(new_card):
            import madgraph.interface.extended_cmd as extended_cmd
            try:
                import internal.extended_cmd as extended_internal
                Shell_internal = extended_internal.CmdShell
            except:
                Shell_internal = None
            if not isinstance(self.mother, (extended_cmd.CmdShell, Shell_internal)): 
                raise Exception("scan are not allowed on the Web")
            # at least one scan parameter found. create an iterator to go trough the cards
            main_card = check_param_card.ParamCardIterator(new_card)
            if self.options['rwgt_name']:
                self.options['rwgt_name'] = '%s_0' % self.options['rwgt_name']

            param_card_iterator = main_card
            first_card = param_card_iterator.next(autostart=True)
            new_card = first_card.write()
            self.new_param_card = first_card
            #first_card.write(pjoin(rw_dir, 'Cards', 'param_card.dat'))  

        # check if "Auto" is present for a width parameter)
        if 'block' not in new_card.lower():
            raise Exception(str(new_card))
        tmp_card = new_card.lower().split('block',1)[1]
        if "auto" in tmp_card: 
            if param_card_iterator:
                first_card.write(pjoin(rw_dir, 'Cards', 'param_card.dat'))
            else:
                ff = open(pjoin(rw_dir, 'Cards', 'param_card.dat'),'w')
                ff.write(new_card)
                ff.close()
                
            self.mother.check_param_card(pjoin(rw_dir, 'Cards', 'param_card.dat'))
            new_card = open(pjoin(rw_dir, 'Cards', 'param_card.dat')).read()

        #misc.sprint(self.banner)
        # Find new tag in the banner and add information if needed
        if 'initrwgt' in self.banner and self.output_type == 'default':
            if 'name=\'mg_reweighting\'' in self.banner['initrwgt']:
                blockpat = re.compile(r'''<weightgroup name=\'mg_reweighting\'\s*weight_name_strategy=\'includeIdInWeightName\'>(?P<text>.*?)</weightgroup>''', re.I+re.M+re.S)
                before, content, after = blockpat.split(self.banner['initrwgt'])
                header_rwgt_other = before + after
                pattern = re.compile('<weight id=\'(?:rwgt_(?P<id>\\d+)|(?P<id2>[_\\w\\-\\.]+))(?P<rwgttype>\\s*|_\\w+)\'>(?P<info>.*?)</weight>', re.S+re.I+re.M)
                mg_rwgt_info = pattern.findall(content)
                maxid = 0
                for k,(i, fulltag, nlotype, diff) in enumerate(mg_rwgt_info):
                    if i:
                        if int(i) > maxid:
                            maxid = int(i)
                        mg_rwgt_info[k] = (i, nlotype, diff) # remove the pointless fulltag tag
                    else:
                        mg_rwgt_info[k] = (fulltag, nlotype, diff) # remove the pointless id tag
                         
                maxid += 1
                rewgtid = maxid
                if self.options['rwgt_name']:
                    #ensure that the entry is not already define if so overwrites it
                    for (i, nlotype, diff) in mg_rwgt_info[:]:
                        for flag in type_rwgt:
                            if 'rwgt_%s' % i == '%s%s' %(self.options['rwgt_name'],flag) or \
                                i == '%s%s' % (self.options['rwgt_name'], flag):
                                    logger.warning("tag %s%s already defines, will replace it", self.options['rwgt_name'],flag)
                                    mg_rwgt_info.remove((i, nlotype, diff))
                                                
            else:
                header_rwgt_other = self.banner['initrwgt'] 
                mg_rwgt_info = []
                rewgtid = 1
        else:
            self.banner['initrwgt']  = ''
            header_rwgt_other = ''
            mg_rwgt_info = []
            rewgtid = 1

        # add the reweighting in the banner information:
        #starts by computing the difference in the cards.
        s_orig = self.banner['slha']
        self.orig_param_card_text = s_orig
        s_new = new_card
        self.new_param_card = check_param_card.ParamCard(s_new.splitlines())
        
        #define tag for the run
        if self.options['rwgt_name']:
            tag = self.options['rwgt_name']
        else:
            tag = str(rewgtid)

        if 'rwgt_info' in self.options and self.options['rwgt_info']:
            card_diff = self.options['rwgt_info']
            for name in type_rwgt:
                mg_rwgt_info.append((tag, name, self.options['rwgt_info']))
        elif not self.second_model and not self.dedicated_path:
            old_param = check_param_card.ParamCard(s_orig.splitlines())
            new_param =  self.new_param_card
            card_diff = old_param.create_diff(new_param)
            if card_diff == '' and not self.second_process:
                    logger.warning(' REWEIGHTING: original card and new card are identical.')
            try:
                if old_param['sminputs'].get(3)- new_param['sminputs'].get(3) > 1e-3 * new_param['sminputs'].get(3):
                    logger.warning("We found different value of alpha_s. Note that the value of alpha_s used is the one associate with the event and not the one from the cards.")
            except Exception as error:
                logger.debug("error in check of alphas: %s" % str(error))
                pass #this is a security                
            if not self.second_process:
                for name in type_rwgt:
                    mg_rwgt_info.append((tag, name, card_diff))
            else:
                str_proc = "\n change process  ".join([""]+self.second_process)
                for name in type_rwgt:
                    mg_rwgt_info.append((tag, name, str_proc + '\n'+ card_diff))
        else:
            if self.second_model:
                str_info = "change model %s" % self.second_model
            else:
                str_info =''
            if self.second_process:
                str_info += "\n change process  ".join([""]+self.second_process)
            if self.dedicated_path:
                for k,v in self.dedicated_path.items():
                    str_info += "\n change %s %s" % (k,v)
            card_diff = str_info
            str_info += '\n' + s_new
            for name in type_rwgt:
                mg_rwgt_info.append((tag, name, str_info))
        # re-create the banner.
        self.banner['initrwgt'] = header_rwgt_other
        if self.output_type == 'default':
            self.banner['initrwgt'] += '\n<weightgroup name=\'mg_reweighting\' weight_name_strategy=\'includeIdInWeightName\'>\n'
        else:
            self.banner['initrwgt'] += '\n<weightgroup name=\'main\'>\n'
        for tag, rwgttype, diff in mg_rwgt_info:
            if tag.isdigit():
                self.banner['initrwgt'] += '<weight id=\'rwgt_%s%s\'>%s</weight>\n' % \
                                       (tag, rwgttype, diff)
            else:
                self.banner['initrwgt'] += '<weight id=\'%s%s\'>%s</weight>\n' % \
                                       (tag, rwgttype, diff)
        self.banner['initrwgt'] += '\n</weightgroup>\n'
        self.banner['initrwgt'] = self.banner['initrwgt'].replace('\n\n', '\n')


        logger.info('starts to compute weight for events with the following modification to the param_card:')
        logger.info(card_diff.replace('\n','\nKEEP:'))
        try:
            self.run_card = banner.Banner(self.banner).charge_card('run_card')
        except Exception:
            logger.debug('no run card found -- reweight interface')
            self.run_card = None

        if self.options['rwgt_name']:
            tag_name = self.options['rwgt_name']
        else:
            tag_name = 'rwgt_%s' % rewgtid

                
        #initialise module.
        for (path,tag), module in self.f2pylib.items():
            with misc.chdir(pjoin(os.path.dirname(rw_dir), path)):
                with misc.stdchannel_redirected(sys.stdout, os.devnull):                    
                    if 'rw_me_' in path or tag == 3:
                        param_card = self.new_param_card
                    else:
                        param_card = check_param_card.ParamCard(self.orig_param_card_text)
                    module.initialise('../Cards/param_card.dat')
                    for block in param_card:
                        if block.lower() == 'qnumbers':
                            continue
                        for param   in param_card[block]:
                            lhacode = param.lhacode
                            value = param.value
                            name = '%s_%s' % (block.upper(), '_'.join([str(i) for i in lhacode]))
                            module.change_para(name, value)
                        if param_card[block].scale:
                            name = "mdl__%s__scale" % block.upper()
                            module.change_para(name, param_card[block].scale)

                    #check for running attribute
                    update_running_info = False
                    if tag == 2:
                        if not self.model:
                            update_running_info = True
                        elif  self.model["running_elements"]:
                            update_running_info = True
                    elif self.second_model:
                        if self.second_model["running_elements"]:
                            update_running_info = True
                    elif  not self.model:
                        update_running_info = True
                    elif self.model["running_elements"]:
                        update_running_info = True
                    if update_running_info:
                        try:
                            run_card = banner.RunCard(self.banner.get('run_card'))
                            module.set_fixed_extra_scale(run_card['fixed_extra_scale'])
                            module.set_mue_over_ref(run_card['mue_over_ref'])
                            module.set_mue_ref_fixed(run_card['mue_ref_fixed'])
                            module.set_maxjetflavor(run_card['maxjetflavor'])
                            module.set_asmz(param_card.get('sminputs').get((3,)).value)
                            module.set_nloop(2)
                        except Exception:
                            if self.model:
                                raise
                    module.update_all_coup()
                        
        return param_card_iterator, tag_name

        
    def do_set(self, line):
        "Not in help"
        
        logger.warning("Invalid Syntax. The command 'set' should be placed after the 'launch' one. Continuing by adding automatically 'launch'")
        self.stored_line = "set %s" % line
        return self.exec_cmd("launch")
    
    def default(self, line, log=True):
        """Default action if line is not recognized"""

        if os.path.isfile(line):
            if log:
                logger.warning("Invalid Syntax. The path to a param_card' should be placed after the 'launch' command. Continuing by adding automatically 'launch'")
            self.stored_line =  line
            return self.exec_cmd("launch")
        else:
            return super(ReweightInterface,self).default(line, log=log)
    
    def write_reweighted_event(self, event, tag_name, **opt):
        """a function for running in multicore"""
        
        if not hasattr(opt['thread_space'], "calculator"):
            opt['thread_space'].calculator = {}
            opt['thread_space'].calculator_nbcall = {}
            opt['thread_space'].cross = 0
            opt['thread_space'].output = open( self.lhe_input.name +'rw.%s' % opt['thread_id'], 'w')
            if self.mother:
                out_path = pjoin(self.mother.me_dir, 'Events', 'reweight.lhe.%s' % opt['thread_id'])
                opt['thread_space'].output2 = open(out_path, 'w')
                
        weight = self.calculate_weight(event, space=opt['thread_space'])
        opt['thread_space'].cross += weight
        if self.output_type == "default":
            event.reweight_data[tag_name] = weight
            #write this event with weight
            opt['thread_space'].output.write(str(event))
            if self.mother:
                event.wgt = weight
                event.reweight_data = {}
                opt['thread_space'].output2.write(str(event))
        else:
            event.wgt = weight
            event.reweight_data = {}
            if self.mother:
                opt['thread_space'].output2.write(str(event))
            else:
                opt['thread_space'].output.write(str(event))
        
        return 0
    
    def do_compute_widths(self, line):
        return self.mother.do_compute_widths(line)
    
    
    dynamical_scale_warning=True
    def change_kinematics(self, event):
 
        if isinstance(self.run_card, banner.RunCardLO):
            jac = event.change_ext_mass(self.new_param_card)
            new_event = event
        else:
            jac =1
            new_event = event

        if jac != 1:
            if self.output_type == 'default':
                logger.critical('mass reweighting requires dedicated lhe output!. Please include "change output 2.0" in your reweight_card')
                raise Exception
            mode = self.run_card['dynamical_scale_choice']
            if mode == -1:
                if self.dynamical_scale_warning:
                    logger.warning('dynamical_scale is set to -1. New sample will be with HT/2 dynamical scale for renormalisation scale')
                mode = 3
            new_event.scale = event.get_scale(mode)
            new_event.aqcd = self.lhe_input.get_alphas(new_event.scale, lhapdf_config=self.mother.options['lhapdf'])
         
        return jac, new_event
    
    
    def calculate_weight(self, event):
        """space defines where to find the calculator (in multicore)"""
        
        #This block imports the PDF information and sends the calculation to the correct function
        if self.has_nlo and self.rwgt_mode != "LO":
            if not hasattr(self,'pdf'):
                lhapdf = misc.import_python_lhapdf(self.mg5cmd.options['lhapdf'])
                self.pdf = lhapdf.mkPDF(self.banner.run_card.get_lhapdf_id())
                
            return self.calculate_nlo_weight(event)
        
        event.parse_reweight()                    
        orig_wgt = event.wgt
        # LO reweighting    


        w_orig = self.calculate_matrix_element(event, 0)
        
        if self.flag_density_matrix: #If we just want to calculate the density we exit the function here
            return {'orig': w_orig}




        # reshuffle event for mass effect # external mass only
        # carefull that new_event can sometimes be = to event 
        # (i.e. change can be in place)
        jac, new_event = self.change_kinematics(event)
        
        
        if event.wgt != 0: # impossible reshuffling
            w_new =  self.calculate_matrix_element(new_event, 1)
        else:
            w_new = 0

        if w_orig == 0:
            tag, order = event.get_tag_and_order()
            orig_order, Pdir, hel_dict = self.id_to_path[tag]
            misc.sprint(w_orig, w_new)
            misc.sprint(event)
            misc.sprint(self.invert_momenta(event.get_momenta(orig_order)))
            misc.sprint(event.get_momenta(orig_order))
            misc.sprint(event.aqcd)
            hel_order = event.get_helicity(orig_order)
            if self.helicity_reweighting and 9 not in hel_order:
                nhel = hel_dict[tuple(hel_order)]
            else:
                nhel = 0
            raise Exception("Invalid matrix element for original computation (weight=0)")

        return {'orig': orig_wgt, '': w_new/w_orig*orig_wgt*jac}
     
    def calculate_nlo_weight(self, event):


        type_nlo = self.get_weight_names()
        final_weight = {'orig': event.wgt}
            
        event.parse_reweight()
        event.parse_nlo_weight(threshold=self.soft_threshold) 
        if not event.nloweight.ispureqcd():
            raise Exception('NLO reweighting does not support mixed expansion mode. Only LO accurate mode is allowed.')
        
        if self.output_type != 'default':
            event.nloweight.modified = True # the internal info will be changed
                                            # so set this flage to True to change
                                            # the writting of those data

        #initialise the input to the function which recompute the weight
        scales2 = []
        pdg = []
        bjx = []
        wgt_tree = [] # reweight for loop-improved type
        wgt_virt  = [] #reweight b+v together
        base_wgt = []
        gs=[]
        qcdpower = []
        ref_wgts = [] #for debugging

        orig_wgt = 0
        for cevent in event.nloweight.cevents:
            #check if we need to compute the virtual for that cevent
            need_V = False # the real is nothing else than the born for a N+1 config
            all_ctype = [w.type for w in cevent.wgts]
            if '_nlo' in type_nlo and any(c in all_ctype for c in [2,14,15]):
                need_V =True
            
            w_orig = self.calculate_matrix_element(cevent, 0)
            w_new =  self.calculate_matrix_element(cevent, 1)
            ratio_T = w_new/w_orig

            if need_V:
                scale2 = cevent.wgts[0].scales2[0]
                #for scale2 in set(c.scales2[1] for c in cevent.wgts): 
                w_origV = self.calculate_matrix_element(cevent, 'V0', scale2=scale2**2)
                w_newV =  self.calculate_matrix_element(cevent, 'V1', scale2=scale2**2)                    
                ratio_BV = (w_newV + w_new) / (w_origV + w_orig)
                ratio_V = w_newV/w_origV if w_origV else  "should not be used"
            else:
                ratio_V = "should not be used"
                ratio_BV = "should not be used"
            for c_wgt in cevent.wgts:
                orig_wgt += c_wgt.ref_wgt
                #add the information to the input
                scales2.append(c_wgt.scales2)
                pdg.append(c_wgt.pdgs[:2])

                bjx.append(c_wgt.bjks)
                qcdpower.append(c_wgt.qcdpower)
                gs.append(c_wgt.gs)
                ref_wgts.append(c_wgt.ref_wgt)
                
                if '_nlo' in type_nlo:
                    if c_wgt.type in  [2,14,15]:
                        R = ratio_BV
                    else:
                        R = ratio_T
                    
                    new_wgt = [c_wgt.pwgt[0] * R,
                               c_wgt.pwgt[1] * ratio_T,
                               c_wgt.pwgt[2] * ratio_T]
                    wgt_virt.append(new_wgt)

                if '_tree' in type_nlo:
                    new_wgt = [c_wgt.pwgt[0] * ratio_T,
                               c_wgt.pwgt[1] * ratio_T,
                               c_wgt.pwgt[2] * ratio_T]
                    wgt_tree.append(new_wgt)
                    
                base_wgt.append(c_wgt.pwgt[:3])
        
        
        orig_wgt_check, partial_check = self.combine_wgt_local(scales2, pdg, bjx, base_wgt, gs, qcdpower, self.pdf)
        #change the ordering to the fortran one: 
        #scales2_i = self.invert_momenta(scales2)
        #pdg_i = self.invert_momenta(pdg)
        #bjx_i = self.invert_momenta(bjx)
        # re-compute original weight to reduce numerical inacurracy
        #base_wgt_i = self.invert_momenta(base_wgt)
        #orig_wgt_check, partial_check = self.combine_wgt(scales2_i, pdg_i, bjx_i, base_wgt_i, gs, qcdpower, 1., 1.)
        
        if '_nlo' in type_nlo:
            #wgt = self.invert_momenta(wgt_virt)
            with misc.stdchannel_redirected(sys.stdout, os.devnull):
                new_out, partial = self.combine_wgt_local(scales2, pdg, bjx, wgt_virt, gs, qcdpower, self.pdf)
            # try to correct for precision issue
            avg = [partial_check[i]/ref_wgts[i] for i in range(len(ref_wgts))]
            out = sum(partial[i]/avg[i] if 0.85<avg[i]<1.15 else 0 \
                          for i in range(len(avg)))
            final_weight['_nlo'] = out/orig_wgt*event.wgt

            
        if '_tree' in type_nlo:
            #wgt = self.invert_momenta(wgt_tree)
            with misc.stdchannel_redirected(sys.stdout, os.devnull):
                out, partial = self.combine_wgt_local(scales2, pdg, bjx, wgt_tree, gs, qcdpower, self.pdf)
            # try to correct for precision issue
            avg = [partial_check[i]/ref_wgts[i] for i in range(len(ref_wgts))]
            new_out = sum(partial[i]/avg[i] if 0.85<avg[i]<1.15 else partial[i] \
                          for i in range(len(avg)))
            final_weight['_tree'] = new_out/orig_wgt*event.wgt    
                  
             
        if '_lo' in type_nlo:
            w_orig = self.calculate_matrix_element(event, 0)
            w_new =  self.calculate_matrix_element(event, 1)            
            final_weight['_lo'] = w_new/w_orig*event.wgt
            
            
        if self.output_type != 'default' and len(type_nlo)==1 and '_lo' not in type_nlo:
            to_write = [partial[i]/ref_wgts[i]*partial_check[i]
                             if 0.85<avg[i]<1.15 else 0
                              for i in range(len(ref_wgts))]
            for cevent in event.nloweight.cevents:
                for c_wgt in cevent.wgts:
                        c_wgt.ref_wgt = to_write.pop(0)
                        if '_tree' in type_nlo:
                            c_wgt.pwgt = wgt_tree.pop(0)
                        else:
                            c_wgt.pwgt = wgt_virt.pop(0)
            assert not to_write
            assert not wgt_tree
        return final_weight 


    def combine_wgt_local(self, scale2s, pdgs, bjxs, base_wgts, gss, qcdpowers, pdf):

        wgt = 0.
        wgts = []
        for (scale2, pdg, bjx, base_wgt, gs, qcdpower) in   zip(scale2s, pdgs, bjxs, base_wgts, gss, qcdpowers):
            Q2, mur2, muf2 = scale2 #Q2 is Ellis-Sexton scale
            #misc.sprint(Q2, mur2, muf2, base_wgt, gs, qcdpower)
            pdf1 = pdf.xfxQ2(pdg[0], bjx[0], muf2)/bjx[0]
            pdf2 = pdf.xfxQ2(pdg[1], bjx[1], muf2)/bjx[1]
            alphas = pdf.alphasQ2(mur2)
            tmp = base_wgt[0] + base_wgt[1] * math.log(mur2/Q2) + base_wgt[2] * math.log(muf2/Q2)
            tmp *= gs**qcdpower*pdf1*pdf2
            wgt += tmp
            wgts.append(tmp)
        return wgt, wgts
        
    
     
    @staticmethod   
    def invert_momenta(p):
        """ fortran/C-python do not order table in the same order"""
        new_p = []
        for i in range(len(p[0])):  new_p.append([0]*len(p))
        for i, onep in enumerate(p):
            for j, x in enumerate(onep):
                new_p[j][i] = x
        return new_p
    
    @staticmethod
    def rename_f2py_lib(Pdir, tag):
        if tag == 2:
            return
        if os.path.exists(pjoin(Pdir, 'matrix%spy.so' % tag)):
            return
        else:
            open(pjoin(Pdir, 'matrix%spy.so' % tag),'w').write(open(pjoin(Pdir, 'matrix2py.so')
                                        ).read().replace('matrix2py', 'matrix%spy' % tag))
    
    def calculate_matrix_element(self, event, hypp_id, scale2=0):
        """routine to return the matrix element or the density matrix"""

        if self.has_nlo:
            nb_retry, sleep = 10, 60 
        else:
            nb_retry, sleep = 5, 20 
        
        tag, order = event.get_tag_and_order()
        #misc.sprint(self.keep_ordering) #I am not sure what is keep_ordering so I print it
        if self.keep_ordering:
            old_tag = tuple(tag)
            tag = (tag[0], tuple(order[1])) 
        if isinstance(hypp_id, str) and hypp_id.startswith('V'):
            tag = (tag,'V')
            hypp_id = int(hypp_id[1:])
        #    base = "rw_mevirt"
        #else:
        #    base = "rw_me"

        if (not self.second_model and not self.second_process and not self.dedicated_path) or hypp_id==0:
            orig_order, Pdir, hel_dict = self.id_to_path[tag]
        else:
            try:
                orig_order, Pdir, hel_dict = self.id_to_path_second[tag]
            except KeyError:
                if self.options['allow_missing_finalstate']:
                    return 0.0
                else:
                    logger.critical('The following initial/final state %s can not be found in the new model/process. If you want to set the weights of such events to zero use "change allow_missing_finalstate False"', tag)
                    raise Exception

        base = os.path.basename(os.path.dirname(Pdir))

        if base == 'rw_me':
            moduletag = (base, 2+hypp_id)
        else:
            moduletag = (base, 2)

        module = self.f2pylib[moduletag]

        if self.keep_ordering:
            all_p = [event.get_momenta(orig_order)]
        else:
            all_p = event.get_all_momenta(orig_order)

            if len(all_p) >1:
                if self.helicity_reweighting:
                    logger.warning("due to ordering ambiguity, we flip off helicity per helicity reweighting.")
                self.helicity_reweighting = False

        # add helicity information
        
        hel_order = event.get_helicity(orig_order)
        if self.helicity_reweighting and 9 not in hel_order:
            nhel = hel_dict[tuple(hel_order)]
        else:
            nhel = -1

        pdg = list(orig_order[0])+list(orig_order[1]) #my code should work if this is the order in the lhe file, see the order 'keep_ordering'

        #list_properties is the list of properties of the class FourMomentum that we can use to ordonate our particles
        list_properties = [p for p in dir(lhe_parser.FourMomentum) if isinstance(getattr(lhe_parser.FourMomentum,p),property)]
        
#######################BOOST BLOCK#############################
        if self.flag_density_matrix:
            if self.momenta_boost[1] == '':
                new_order_momenta = tuple([i for i in range(len(event))])
            else:
                found_property_boost = False
                for prop in list_properties:
                    if prop in self.momenta_boost[1]:
                        found_property_boost = True
                        information_lhe_order = []
                        original_lhe_order = [i for i in range(len(event))]
                        for i, p in enumerate(event):
                            if pdg[i] in self.momenta_boost[0]:
                                correct_p = lhe_parser.FourMomentum(p)
                                information_lhe_order.append(getattr(correct_p, prop))
                            else:
                                information_lhe_order.append(float('NaN'))

                        #Check if there are dupplicate in the list
                        if len(set(information_lhe_order)) != len(information_lhe_order):
                            logger.warning("Some particles in the boosting option have the same value for the chosen observable. Their ordering is random. Please ensure to choose an obsevarble that allows to discrimate all the identical particles.")
                        
                        information_lhe_order_momenta, new_order_momenta = zip(*sorted(zip(information_lhe_order, original_lhe_order), reverse=True)) #values of the observable ordered
                        
                        if len(self.momenta_boost[2]) > 0: #this block allows to take into account the choice of order by the user

                            order_input_boost = self.momenta_boost[2]
                            information_lhe_order_momenta_without_nan = [elem for elem in information_lhe_order_momenta if elem == elem]
                            observable_order_momenta = []
                            
                            #This bloc reorders the values of the observarble according to the user's input
                            for i in range(len(order_input_boost)):
                                observable_order_momenta.append(information_lhe_order_momenta_without_nan[order_input_boost[i]])

                            order_input_momenta_sorted = sorted(order_input_boost, reverse=True)
                            for i in range(len(order_input_momenta_sorted)):
                                information_lhe_order_momenta_without_nan.pop(order_input_momenta_sorted[i])

                            for k in range(len(information_lhe_order_momenta_without_nan)):
                                observable_order_momenta.append(information_lhe_order_momenta_without_nan[k])

                            input_information_lhe_order_momenta = [0] * len(information_lhe_order_momenta)
                            input_compteur_momenta = 0
                            for i in range(len(information_lhe_order_momenta)):
                                if information_lhe_order_momenta[i] == information_lhe_order_momenta[i]:
                                    input_information_lhe_order_momenta[i] = observable_order_momenta[input_compteur_momenta]
                                    input_compteur_momenta += 1
                                else:
                                    input_information_lhe_order_momenta[i] = float("NaN")
                            
                            order_input_boost = [0] * len(information_lhe_order_momenta)
                            for i in range(len(input_information_lhe_order_momenta)):
                                if input_information_lhe_order_momenta[i] != input_information_lhe_order_momenta[i]:
                                    order_input_boost[i] = i
                                else:
                                    for j in range(len(information_lhe_order)):
                                        if input_information_lhe_order_momenta[i] == information_lhe_order[j]:
                                            order_input_boost[i] = j

                            pos_aux_momenta = self.momenta_boost[0]

                            boost_corrected = [0] * len(pos_aux_momenta)
                            is_particle_taken_good_momenta = [0] * len(pdg)
                            compteur_momenta_good = 0
                            for i in range(len(pos_aux_momenta)):
                                 for j in range(len(pdg)):
                                    if pdg[order_input_boost[j]] == pos_aux_momenta[i] and is_particle_taken_good_momenta[j] == 0:
                                        boost_corrected[compteur_momenta_good] = order_input_boost[j]  #no +1 because python format
                                        is_particle_taken_good_momenta[j] = 1
                                        compteur_momenta_good += 1
                                        break
                            #result of this block: boost_corrected

                if not found_property_boost:
                    new_order_momenta = tuple([i for i in range(len(event))])
                    logger.warning("The observable given as input is not defined in the FourMomentum class. using default order.")

            if len(self.momenta_boost[2]) == 0:
                if -1 not in self.momenta_boost[0]:
                    boost_corrected = [0] * len(self.momenta_boost[0])
                    is_particle_taken_boost = [0] * len(pdg)
                    compteur_boost = 0
                    for i in range(len(boost_corrected)):
                        for j in range(len(pdg)):
                            if pdg[new_order_momenta[j]] == self.momenta_boost[0][i] and is_particle_taken_boost[j] == 0:
                                boost_corrected[compteur_boost] = new_order_momenta[j] # no +1 because python format
                                is_particle_taken_boost[j] = 1
                                compteur_boost += 1
                                break

                    #result of this block: boost_corrected
                    
                else:
                    boost_corrected = [-1]
            

        #Here we call the boost function, there is a slight difference between the reweight/density mode that needs to separate it in 2 cases
        if self.flag_density_matrix:
            all_p = self.boost_event_density(event, all_p, orig_order, hypp_id, boost_corrected)
        else:
            all_p = self.boost_event_density(event, all_p, orig_order, hypp_id)

#######################END BOOST BLOCK#############################


#######################ROTATION CHOICE BLOCK#############################
        if self.flag_density_matrix: #this big block allows to define the orders in which to chose the PDG-codes when there are several identical in the process
            if self.helicity_direction[1] == '':
                new_order_rot = tuple([i for i in range(len(event))])

            else:
                found_property_rot = False
                for prop in list_properties:
                    if prop in self.helicity_direction[1]:
                        found_property_rot = True
                        information_lhe_order = []
                        original_lhe_order = [i for i in range(len(event))]
                        for i, p in enumerate(event):
                            if pdg[i] in self.helicity_direction[0]:
                                correct_p = lhe_parser.FourMomentum(p)
                                information_lhe_order.append(getattr(correct_p, prop))
                            else:
                                information_lhe_order.append(float('NaN'))

                        if len(set(information_lhe_order)) != len(information_lhe_order):
                            logger.warning("Some particles in the helicity direction option have the same value for the chosen observable. Their ordering is random. Please ensure to choose an obsevarble that allows to discrimate all the identical particles.")
                        
                        new_information_lhe_rot, new_order_rot = zip(*sorted(zip(information_lhe_order, original_lhe_order), reverse=True))

                        if len(self.helicity_direction[2]) > 0: #this block allows to take into account the choice of order by the user

                            order_input_rot = self.helicity_direction[2]
                            information_lhe_order_without_nan_rot = [elem for elem in new_information_lhe_rot if elem == elem]
                            observable_order_rot = []
                            
                            for i in range(len(order_input_rot)):
                                observable_order_rot.append(information_lhe_order_without_nan_rot[order_input_rot[i]] )

                            order_input_sorted_rot = sorted(order_input_rot, reverse=True)
                            for i in range(len(order_input_sorted_rot)):
                                information_lhe_order_without_nan_rot.pop(order_input_sorted_rot[i])

                            for k in range(len(information_lhe_order_without_nan_rot)):
                                observable_order_rot.append(information_lhe_order_without_nan_rot[k])

                            input_information_lhe_order_rot = [0] * len(new_information_lhe_rot)
                            input_compteur_rot = 0
                            for i in range(len(new_information_lhe_rot)):
                                if new_information_lhe_rot[i] == new_information_lhe_rot[i]:
                                    input_information_lhe_order_rot[i] = observable_order_rot[input_compteur_rot]
                                    input_compteur_rot += 1
                                else:
                                    input_information_lhe_order_rot[i] = float("NaN")
                            
                            good_order_rot = [0] * len(information_lhe_order)
                            for i in range(len(input_information_lhe_order_rot)):
                                if input_information_lhe_order_rot[i] != input_information_lhe_order_rot[i]:
                                    good_order_rot[i] = i
                                else:
                                    for j in range(len(information_lhe_order)):
                                        if input_information_lhe_order_rot[i] == information_lhe_order[j]:
                                            good_order_rot[i] = j

                            pos_aux = self.helicity_direction[0]
                            pos_good_rot = [0] * len(pos_aux)
                            is_particle_taken_good_rot = [0] * len(pdg)
                            compteur_pos_good_rot = 0
                            for i in range(len(pos_aux)):
                                for j in range(len(pdg)):
                                    if pdg[good_order_rot[j]] == pos_aux[i] and is_particle_taken_good_rot[j] == 0:
                                        pos_good_rot[compteur_pos_good_rot] = good_order_rot[j] + 1 #+1 because fortran format
                                        is_particle_taken_good_rot[j] = 1
                                        compteur_pos_good_rot += 1
                                        break
                            refChoice_corrected = pos_good_rot

                if not found_property_rot:
                    new_order_rot = tuple([i for i in range(len(event))])
                    logger.warning("The observable given as input is not defined in the FourMomentum class. Using default order.")

            if len(self.helicity_direction[2]) == 0:
                refChoice = [0] * len(self.helicity_direction[0])
                if -1 in self.helicity_direction[0]: #all the angles are set to 0 and the rotation is done
                    phi, theta = [0] * len(all_p), [0] * len(all_p)
                    for i in range(len(all_p)):
                        all_p[i] = self.invert_momenta(all_p[i])
                        all_p[i] = module.rotationp(all_p[i], phi[i], theta[i], len(pdg))
                        all_p[i] = self.invert_momenta(all_p[i])
                        for j in range(len(all_p[i])):
                            all_p[i][j] = tuple(all_p[i][j])
                    refChoice_corrected = [-1]

                else:
                    is_particle_taken_refChoice = [0] * len(pdg)
                    compteur_refChoice = 0
                    input_pdg = self.helicity_direction[0]
                    for i in range(len(refChoice)):
                        for j in range(len(pdg)):
                            if pdg[new_order_rot[j]] == input_pdg[i] and is_particle_taken_refChoice[new_order_rot[j]] == 0:
                                refChoice[compteur_refChoice] = new_order_rot[j] + 1 #+1 because fortran format
                                is_particle_taken_refChoice[new_order_rot[j]] = 1
                                compteur_refChoice += 1
                                break #needed to put only one particle for each refChoice[i]

                    refChoice_corrected = refChoice

            if -1 not in refChoice_corrected:
                pref = [0, 0, 0, 0]
                phi, theta = [0] * len(all_p), [0] * len(all_p)
                for i in range(len(all_p)):
                    for j in range(len(refChoice_corrected)):
                        for k in range(len(pref)):
                            pref[k] += all_p[i][refChoice_corrected[j] - 1][k]
                    phi[i], theta[i] = module.refchoicep(pref) #angles phi and theta are defined here

                    #########
                    #This block allows to choose which initial state particle is chosen as reference to define theta.
                    #If its pz is > 0 the default definition is correct, if it is < 0, then we need to add pi

                    if self.axis_referential:
                        for k in range(len(self.axis_referential)):
                            if self.axis_referential[k] in orig_order[0]: #check whether the pdg is in the initial state, if it is not we do not change anything

                                for j in range(len(orig_order[0])):
                                    if self.axis_referential[k] == orig_order[0][j]:
                                        pz_axis_referential = all_p[i][j][3] #here we take the p_z of the chosen particle
                                if pz_axis_referential < 0:
                                    theta[i] += math.pi

                    if self.symmetrise_initial_state: #if we want to calculate R(theta) + R(theta + pi)
                        import copy
                        all_p_bis = copy.deepcopy(all_p)
                        all_p_bis[i] = self.invert_momenta(all_p_bis[i]) #put in fortran format
                        all_p_bis[i] = module.rotationp(all_p_bis[i], phi[i], theta[i] + math.pi, len(pdg))
                        all_p_bis[i] = self.invert_momenta(all_p_bis[i]) #put back into python format
                
                    #########
                    all_p[i] = self.invert_momenta(all_p[i]) #put in fortran format
                    all_p[i] = module.rotationp(all_p[i], phi[i], theta[i], len(pdg))
                    all_p[i] = self.invert_momenta(all_p[i]) #put back into python format

                    for j in range(len(all_p[i])):
                        all_p[i][j] = tuple(all_p[i][j]) #we put the momenta back into tuples because it was structured like that initially

            if self.options['identical_particle_in_prod_and_decay'] == 'crash':
                if len(all_p) > 1:
                    raise Exception("Ambiguous particle in production and decay. crash as requested by \'identical_particle_in_prod_and_decay\'")
            
#######################END ROTATION CHOICE BLOCK#############################


#######################PARTICLE CHOICE BLOCK#############################
        if self.flag_density_matrix:
            if self.particle_in_density_matrix[1] == '':
                new_order_pos = tuple([i for i in range(len(event))])
                new_pdg_pos = [0] * len(pdg)
                for k in range(len(pdg)):
                    new_pdg_pos[k] = pdg[new_order_pos[k]]
            else:
                found_property = False
                for prop in list_properties:
                    if prop in self.particle_in_density_matrix[1]:
                        found_property = True
                        information_lhe_order = []
                        original_lhe_order = [i for i in range(len(event))]
                        for i, p in enumerate(event):
                            if pdg[i] in self.particle_in_density_matrix[0]:
                                correct_p = lhe_parser.FourMomentum(p)
                                information_lhe_order.append(getattr(correct_p, prop))
                            else:
                                information_lhe_order.append(float('NaN'))

                        if len(set(information_lhe_order)) != len(information_lhe_order):
                            logger.warning("Some particles in the density matrix have the same value for the chosen observable. Their ordering is random. Please ensure to choose an obsevarble that allows to discrimate all the identical particles.")

                        new_information_lhe_order, new_order_pos = zip(*sorted(zip(information_lhe_order, original_lhe_order), reverse=True))

                        if len(self.particle_in_density_matrix[2]) > 0: #this block allows to take into account the choice of order by the user

                            order_input = self.particle_in_density_matrix[2]
                            information_lhe_order_without_nan = [elem for elem in new_information_lhe_order if elem == elem]
                            observable_order = []
                            
                            #This bloc reorders the values of the obsevarble according to the user's input
                            for i in range(len(order_input)):
                                observable_order.append(information_lhe_order_without_nan[order_input[i]])

                            #We need to order the input because of the pop method
                            order_input_sorted = sorted(order_input, reverse=True)
                            for i in range(len(order_input_sorted)):
                                information_lhe_order_without_nan.pop(order_input_sorted[i])

                            for k in range(len(information_lhe_order_without_nan)):
                                observable_order.append(information_lhe_order_without_nan[k])

                            input_information_lhe_order = [0] * len(new_information_lhe_order)
                            input_compteur = 0
                            for i in range(len(new_information_lhe_order)):
                                if new_information_lhe_order[i] == new_information_lhe_order[i]:
                                    input_information_lhe_order[i] = observable_order[input_compteur]
                                    input_compteur += 1
                                else:
                                    input_information_lhe_order[i] = float("NaN")

                            good_order = [0] * len(information_lhe_order)
                            for i in range(len(input_information_lhe_order)):
                                if input_information_lhe_order[i] != input_information_lhe_order[i]:
                                    good_order[i] = i
                                else:
                                    for j in range(len(information_lhe_order)):
                                        if input_information_lhe_order[i] == information_lhe_order[j]:
                                            good_order[i] = j

                            pos_aux = self.particle_in_density_matrix[0]

                            pos_good = [0] * len(pos_aux)
                            is_particle_taken_good = [0] * len(pdg)
                            compteur_pos_good = 0
                            for i in range(len(pos_aux)):
                                for j in range(len(pdg)):
                                    if pdg[good_order[j]] == pos_aux[i] and is_particle_taken_good[j] == 0:
                                        pos_good[compteur_pos_good] = good_order[j] + 1 #+1 because fortran format
                                        is_particle_taken_good[j] = 1
                                        compteur_pos_good += 1
                                        break
                            pos_corrected = pos_good

                if not found_property:
                    new_order_pos = tuple([i for i in range(len(event))])

            if len(self.particle_in_density_matrix[2]) == 0:
                pos_aux = self.particle_in_density_matrix[0]

                pos_new = [0] * len(pos_aux)
                is_particle_taken_new = [0] * len(pdg)
                compteur_pos_new = 0
                for i in range(len(pos_aux)):
                    for j in range(len(pdg)):
                        if pdg[new_order_pos[j]] == pos_aux[i] and is_particle_taken_new[j] == 0:
                            pos_new[compteur_pos_new] = new_order_pos[j] + 1 #+1 because fortran format
                            is_particle_taken_new[j] = 1 ################Is it correct ?
                            compteur_pos_new += 1
                            break #needed to put only one particle for each pos_aux[i]

                pos_corrected = pos_new 

#######################END PARTICLE CHOICE BLOCK#############################

        
        if self.flag_density_matrix:
            import madgraph.various.Density_functions as dens
            import numpy as np
            
            status = []
            for particle in event:
                status.append(int(particle.status))


            PDGs, _ = module.get_pdg_order()
            PREFIX = module.get_prefix()
            prefix_cor = []
            All_PDGs = []
            prefix_unique = []
                
            #Block to determine which sets of pdg-codes corresponds to which prefix
            for i in range(len(PREFIX)):
                prefix_cor.append(PREFIX[i].decode('UTF-8').strip().lower())
                if prefix_cor[i] not in prefix_unique:
                    prefix_unique.append(prefix_cor[i])
            for i in range(len(PDGs)):
                All_PDGs.append(dens.permutations_PGD(PDGs[i], status))

            #We take the card in the general folder, not in the reweight folder
            Card_dir = os.path.join(self.me_dir, "Cards", "param_card.dat")

            Initialise_allmatrix = getattr(module, 'initialise')
            Initialise_allmatrix(Card_dir)
            for i in range(len(prefix_unique)):
                InitialiseMatrix = getattr(module, prefix_unique[i] + 'initialisemodel')
                InitialiseMatrix(Card_dir)   

            #The prefix is defined for a given event
            for k in range(len(All_PDGs)):
                    if pdg in All_PDGs[k]:
                        prefix = prefix_cor[k]

            me_value = 0
            for p in all_p:
                pinv = self.invert_momenta(p)
                get_density = getattr(module, prefix + 'get_density')
                new_value = get_density(pinv, pos_corrected, self.number_changing_helicities,
                                        self.allowed_helicities, self.number_combinations,
                                        event.aqcd)

                new_value = dens.get_list_sliced(new_value, self.number_combinations, epsilon=1e-10) #new value is the production matrix, not the density matrix
                #new_value = dens.get_rho_normalised(new_value, self.number_combinations, epsilon=1e-10)


                #I am not sure this block for loop is necessary or not
                # for loop we have also the stability status code
                if isinstance(new_value, tuple):
                    new_value, code = new_value
                    #if code points unstability -> returns 0
                    hundred_value = (code % 1000) //100
                    if hundred_value in [4]:
                        new_value = 0.
                if self.options["identical_particle_in_prod_and_decay"] == "average":
                    me_value += new_value
                elif self.options["identical_particle_in_prod_and_decay"] == "max":
                    if abs(new_value) > abs(me_value):
                        me_value = new_value
                else: 
                    raise Exception("not valid option")

            if self.options["identical_particle_in_prod_and_decay"] == "average":
                return me_value / len(all_p)        
            else:
                return new_value
                
        else:
            me_value = 0
            for p in all_p:
                pold = list(p)
                p = self.invert_momenta(p)
                pdg = list(orig_order[0])+list(orig_order[1])
                try:
                    pid = event.ievent
                except AttributeError:
                    pid = -1
                if not self.use_eventid:
                    pid = -1
                
                if not scale2: 
                    if hasattr(event, 'scale'):
                        scale2 = event.scale**2
                    else:
                        scale2 = 0

                with misc.chdir(Pdir): #we enter the directory Pdir
                    with misc.stdchannel_redirected(sys.stdout, os.devnull):
                        new_value = module.smatrixhel(pdg, pid, p, event.aqcd, scale2, nhel)

                # for loop we have also the stability status code
                if isinstance(new_value, tuple):
                    new_value, code = new_value
                    #if code points unstability -> returns 0
                    hundred_value = (code % 1000) //100
                    if hundred_value in [4]:
                        new_value = 0.
                if self.options["identical_particle_in_prod_and_decay"] == "average":
                    me_value += new_value
                elif self.options["identical_particle_in_prod_and_decay"] == "max":
                    if abs(new_value) > abs(me_value):
                        me_value = new_value
                else: 
                    raise Exception("not valid option")

            if self.options["identical_particle_in_prod_and_decay"] == "average":
                return me_value / len(all_p)        
            else:
                return me_value


    def boost_event_density(self, event, all_p, orig_order, hypp_id, boost_corrected=None):
        # For 2>N pass in the center of mass frame
        #   - required for helicity by helicity re-weighitng
        #   - Speed-up loop computation 
        # the option boost_corrected is only necessary for the density mode

        if self.flag_density_matrix == True: #boost in the case of the density mode
            if -1 in self.momenta_boost[0]: #if we don't want to boost the system
                return all_p
            import copy
            new_event = copy.deepcopy(event)
            nb_ext = 0
            pboost = lhe_parser.FourMomentum()
            for p in new_event: 
                for j in range(len(boost_corrected)):
                    if nb_ext == boost_corrected[j]:
                        pboost += p
                nb_ext += 1


            if abs(pboost.px/pboost.E) < 1e-10 and abs(pboost.py/pboost.E) < 1e-10 and abs(pboost.pz/pboost.E) < 1e-10:
                #if we try to boost with with a 4-momentum like [M, 0, 0, 0], we return the momenta without any boost
                return all_p
            new_event.boost(pboost)
            if self.keep_ordering:
                new_all_p = [new_event.get_momenta(orig_order)]
            else:
                new_all_p = new_event.get_all_momenta(orig_order)
            if len(new_all_p) > 1:
                logger.critical("due to ordering ambiguity, the boost used might not be consistent. please ensure that this is not an issue")

            return new_all_p
        
        
        if (hypp_id == 0 and ('frame_id' in self.banner.run_card and self.banner.run_card['frame_id'] !=6)):
            import copy
            new_event = copy.deepcopy(event)
            pboost = FourMomenta()
            to_inc = bin(self.banner.run_card['frame_id'])[2:]
            to_inc.reverse()
            nb_ext = 0
            for p in new_event:
                if p.status in [-1,1]:
                    nb_ext += 1
                    if to_inc[nb_ext]:
                        pboost += p                    
            new_event.boost(pboost)
            if self.keep_ordering:
                new_all_p = [new_event.get_momenta(orig_order)]
            else:
                new_all_p = new_event.get_all_momenta(orig_order)
            if len(new_all_p) > 1:
                logger.critical("due to ordering ambiguity, the boost used might not be consistent. please ensure that this is not an issue")
                
            return new_all_p

        elif (hypp_id == 1 and self.boost_event):
            if self.boost_event is not True:
                import copy
                new_event = copy.deepcopy(event)
                new_event.boost(self.boost_event)
                if self.keep_ordering:
                    new_all_p = [new_event.get_momenta(orig_order)]
                else:     
                    new_all_p = new_event.get_all_momenta(orig_order)

                return new_all_p
            return all_p #if we arrive here, we should return the input no ?

        elif (hasattr(event[1], 'status') and event[1].status == -1) or \
           (event[1].px == event[1].py == 0.):
            p = all_p[0]
            pboost = lhe_parser.FourMomentum(p[0]) + lhe_parser.FourMomentum(p[1])
            for p in all_p:
                for i,thisp in enumerate(p):
                    p[i] = lhe_parser.FourMomentum(thisp).zboost(pboost).get_tuple()
                assert p[0][1] == p[0][2] == 0 == p[1][2] == p[1][2] == 0 
            
            return all_p
        
        else:
            return all_p

    def terminate_fortran_executables(self, new_card_only=False):
        """routine to terminate all fortran executables"""

        for (mode, production) in dict(self.calculator):
            
            if new_card_only and production == 0:
                continue            
            del self.calculator[(mode, production)]
    
    def do_quit(self, line):
        if self.exitted:
            return
        self.exitted = True
        
        if 'init' in self.banner:
            cross = 0 
            error = 0
            for line in self.banner['init'].split('\n'):
                split = line.split()
                if len(split) == 4:
                    cross, error = float(split[0]), float(split[1])
                    
        if not self.multicore == 'create':
            # No print of results for the multicore mode for the one printed on screen
            if self.flag_density_matrix:
                import madgraph.various.Density_functions as dens
                logger.info("Cross-section: %s +- %s pb" % (cross, error))
                rho_avg = dens.get_rho_normalised(self.production_matrix, self.number_combinations, epsilon=1e-10)
                # for elem in range(len(rho_avg)):
                #     if rho_avg[elem].imag == 0:
                #         rho_avg[elem] = float(rho_avg[elem].real)
                # misc.sprint(rho_avg)
                rho_avg_square = dens.square_matrix(rho_avg)

                logger.info("Average density matrix:")
                for i in range(len(rho_avg_square)):
                    print("\t",list(rho_avg_square[i]))
                #Ca fonctionne mais il faudrait améliorer la présentation
            else:
                if 'orig' not in self.all_cross_section:
                    logger.info('Original cross-section: %s +- %s pb' % (cross, error))
                else: 
                    misc.sprint(self.all_cross_section)
                    logger.info('Original cross-section: %s +- %s pb (cross-section from sum of weights: %s)' % (cross, error, self.all_cross_section['orig'][0]))
                logger.info('Computed cross-section:')
                keys = list(self.all_cross_section.keys())
                keys.sort(key=lambda x: str(x))
                for key in keys:
                    if key == 'orig':
                        continue
                    logger.info('%s : %s +- %s pb' % (key[0] if not key[1] else '%s%s' % key,
                        self.all_cross_section[key][0],self.all_cross_section[key][1] ))  

        self.terminate_fortran_executables()
    
        if self.rwgt_dir and self.multicore == False:
            self.save_to_pickle()
        
        with misc.stdchannel_redirected(sys.stdout, os.devnull):
            for run_id in self.calculator:
                del self.calculator[run_id]
            del self.calculator
        
            
    def __del__(self):
        self.do_quit('')

    
    def adding_me(self, matrix_elements, path):
        """Adding one element to the list based on the matrix element"""
        

    @misc.mute_logger()
    def create_standalone_tree_directory(self, data ,second=False):
        """generate the various directory for the weight evaluation"""
        
        mgcmd = self.mg5cmd         
        path_me = data['path'] 
        # 2. compute the production matrix element -----------------------------
        has_nlo = False  
        mgcmd.exec_cmd("set group_subprocesses False")

        if not second:
            logger.info('generating the square matrix element for reweighting')
        else:
            logger.info('generating the square matrix element for reweighting (second model and/or processes)')
        start = time.time()
        commandline=''
        for i,proc in enumerate(data['processes']):
            if '[' not in proc:
                commandline += "add process %s ;" % proc
            else:
                has_nlo = True
                if self.banner.get('run_card','ickkw') == 3:
                    if len(proc) == min([len(p.strip()) for p in data['processes']]):
                        commandline += self.get_LO_definition_from_NLO(proc, self.model)
                    else:
                        commandline += self.get_LO_definition_from_NLO(proc,
                                                     self.model, real_only=True)
                else:
                    commandline += self.get_LO_definition_from_NLO(proc, self.model)
        
        commandline = commandline.replace('add process', 'generate',1)
        logger.info(commandline)
        try:
            mgcmd.exec_cmd(commandline, precmd=True, errorhandling=False)
        except diagram_generation.NoDiagramException:
            commandline=''
            for proc in data['processes']:
                if '[' not in proc:
                    raise
                # pass to virtsq=
                base, post = proc.split('[',1)
                nlo_order, post = post.split(']',1)
                if '=' not in nlo_order:
                    nlo_order = 'virt=%s' % nlo_order
                elif 'noborn' in nlo_order:
                    nlo_order = nlo_order.replace('noborn', 'virt')
                commandline += "add process %s [%s] %s;" % (base,nlo_order,post)
            commandline = commandline.replace('add process', 'generate',1)
            if commandline:
                logger.info("RETRY with %s", commandline)
                mgcmd.exec_cmd(commandline, precmd=True)
                has_nlo = False
        except Exception as error:
            misc.sprint(type(error))
            raise
        
        commandline = 'output standalone_rw %s --prefix=int' % pjoin(path_me,data['paths'][0])
        mgcmd.exec_cmd(commandline, precmd=True)        
        logger.info('Done %.4g' % (time.time()-start))
        self.has_standalone_dir = True
        

        # 3. Store id to directory information ---------------------------------
        if False:
            # keep this for debugging
            matrix_elements = mgcmd._curr_matrix_elements.get_matrix_elements()
            
            to_check = [] # list of tag that do not have a Pdir at creation time.
            for me in matrix_elements:
                for proc in me.get('processes'):
                    initial = []    #filled in the next line
                    final = [l.get('id') for l in proc.get('legs')\
                          if l.get('state') or initial.append(l.get('id'))]
                    order = (initial, final)
                    tag = proc.get_initial_final_ids()
                    decay_finals = proc.get_final_ids_after_decay()
    
                    if tag[1] != decay_finals:
                        order = (initial, list(decay_finals))
                        decay_finals.sort()
                        tag = (tag[0], tuple(decay_finals))
                    Pdir = pjoin(path_me, data['paths'][0], 'SubProcesses', 
                                      'P%s' % me.get('processes')[0].shell_string())
    
                    if not os.path.exists(Pdir):
                        to_check.append(tag)
                        continue                        
                    if tag in data['id2path']:
                        if not Pdir == data['id2path'][tag][1]:
                            misc.sprint(tag, Pdir, data['id2path'][tag][1])
                            raise self.InvalidCmd('2 different process have the same final states. This module can not handle such situation')
                        else:
                            continue
                    # build the helicity dictionary
                    hel_nb = 0
                    hel_dict = {9:0} # unknown helicity -> use full ME
                    for helicities in me.get_helicity_matrix():
                        hel_nb +=1 #fortran starts at 1
                        hel_dict[tuple(helicities)] = hel_nb
    
                    data['id2path'][tag] = [order, Pdir, hel_dict]        
     
            for tag in to_check:
                if tag not in self.id_to_path:
                    logger.warning("no valid path for %s" % (tag,))
                    #raise self.InvalidCmd, "no valid path for %s" % (tag,)
        
        # 4. Check MadLoopParam for Loop induced
        if os.path.exists(pjoin(path_me, data['paths'][0], 'Cards', 'MadLoopParams.dat')):
            MLCard = banner.MadLoopParam(pjoin(path_me, data['paths'][0], 'Cards', 'MadLoopParams.dat'))
            MLCard.set('WriteOutFilters', False)
            MLCard.set('UseLoopFilter', False)
            MLCard.set("DoubleCheckHelicityFilter", False)
            MLCard.set("HelicityFilterLevel", 0)
            MLCard.write(pjoin(path_me, data['paths'][0], 'SubProcesses', 'MadLoopParams.dat'),
                         pjoin(path_me, data['paths'][0], 'Cards', 'MadLoopParams.dat'), 
                         commentdefault=False)
            
            #if self.multicore == 'create':
            #    print "compile OLP", data['paths'][0]
            #    misc.compile(['OLP_static'], cwd=pjoin(path_me, data['paths'][0],'SubProcesses'),
            #                 nb_core=self.mother.options['nb_core'])
        
        if os.path.exists(pjoin(path_me, data['paths'][1], 'Cards', 'MadLoopParams.dat')):
            if self.multicore == 'create':
                print("compile OLP", data['paths'][1])
                # It is potentially unsafe to use several cores, We limit ourself to one for now
                # n_cores = self.mother.options['nb_core']
                n_cores = 1
                misc.compile(['OLP_static'], cwd=pjoin(path_me, data['paths'][1],'SubProcesses'),
                             nb_core=self.mother.options['nb_core'])
                
        return has_nlo

                
    @misc.mute_logger()
    def create_standalone_virt_directory(self, data ,second=False):
        """generate the various directory for the weight evaluation"""
                
        mgcmd = self.mg5cmd
        path_me = data['path'] 
        # Do not pass here for LO/NLO_tree
        start = time.time()
        commandline=''
        for proc in data['processes']:
            if '[' not in proc:
                pass
            else:
                proc = proc.replace('[', '[ virt=')
                commandline += "add process %s ;" % proc
        commandline = re.sub(r'@\s*\d+', '', commandline)
        # deactivate golem since it creates troubles
        old_options = dict(mgcmd.options)
        if mgcmd.options['golem']:
            logger.info(" When doing NLO reweighting, MG5aMC cannot use the loop reduction algorithms Golem")
        mgcmd.options['golem'] = None            
        commandline = commandline.replace('add process', 'generate',1)
        logger.info(commandline)
        mgcmd.exec_cmd(commandline, precmd=True)
        commandline = 'output standalone_rw %s --prefix=int -f' % pjoin(path_me, data['paths'][1])
        mgcmd.exec_cmd(commandline, precmd=True) 
        
        #put back golem to original value
        mgcmd.options['golem'] = old_options['golem']
        # update make_opts

        if not mgcmd.options['lhapdf']:
            raise Exception("NLO reweighting requires LHAPDF to work correctly")

        # Download LHAPDF SET
        common_run_interface.CommonRunCmd.install_lhapdf_pdfset_static(\
            mgcmd.options['lhapdf'], None, self.banner.run_card.get_lhapdf_id())
        
        # now store the id information             
        if False:
            # keep it for debugging purposes
            matrix_elements = mgcmd._curr_matrix_elements.get_matrix_elements()            
            for me in matrix_elements:
                for proc in me.get('processes'):
                    initial = []    #filled in the next line
                    final = [l.get('id') for l in proc.get('legs')\
                          if l.get('state') or initial.append(l.get('id'))]
                    order = (initial, final)
                    tag = proc.get_initial_final_ids()
                    decay_finals = proc.get_final_ids_after_decay()
    
                    if tag[1] != decay_finals:
                        order = (initial, list(decay_finals))
                        decay_finals.sort()
                        tag = (tag[0], tuple(decay_finals))
                    Pdir = pjoin(path_me, data['paths'][1], 'SubProcesses', 
                                      'P%s' % me.get('processes')[0].shell_string())
                    assert os.path.exists(Pdir), "Pdir %s do not exists" % Pdir                        
                    if (tag,'V') in data['id2path']:
                        if not Pdir == data['id2path'][(tag,'V')][1]:
                            misc.sprint(tag, Pdir, self.id_to_path[(tag,'V')][1])
                            raise self.InvalidCmd('2 different process have the same final states. This module can not handle such situation')
                        else:
                            continue
                    # build the helicity dictionary
                    hel_nb = 0
                    hel_dict = {9:0} # unknown helicity -> use full ME
                    for helicities in me.get_helicity_matrix():
                        hel_nb +=1 #fortran starts at 1
                        hel_dict[tuple(helicities)] = hel_nb
    
                    data['id2path'][(tag,'V')] = [order, Pdir, hel_dict]


    @misc.mute_logger()
    def create_standalone_directory(self, second=False):
        """generate the various directory for the weight evaluation"""
                
        data={}
        if not second:
            data['paths'] = ['rw_me', 'rw_mevirt']
            # model
            info = self.banner.get('proc_card', 'full_model_line')
            if '-modelname' in info:
                data['mg_names'] = False
            else:
                data['mg_names'] = True
            data['model_name'] = self.banner.get('proc_card', 'model')
            #processes
            data['processes'] = [line[9:].strip() for line in self.banner.proc_card
                     if line.startswith('generate')]
            data['processes'] += [' '.join(line.split()[2:]) for line in self.banner.proc_card
                      if re.search(r'^\s*add\s+process', line)]  
            #object_collector
            #self.id_to_path = {}
            #data['id2path'] = self.id_to_path
        else:
            for key in list(self.f2pylib.keys()):
                if 'rw_me_%s' % self.nb_library in key[0]:
                    del self.f2pylib[key]
                
            self.nb_library += 1
            data['paths'] = ['rw_me_%s' % self.nb_library, 'rw_mevirt_%s' % self.nb_library]


            # model
            if self.second_model:
                data['mg_names'] = True
                if ' ' in self.second_model:
                    args = self.second_model.split()
                    if '--modelname' in args:
                        data['mg_names'] = False
                    data['model_name'] = args[0]
                else:
                    data['model_name'] = self.second_model
            else:
                data['model_name'] = None
            #processes
            if self.second_process:
                data['processes'] = self.second_process
            else:
                data['processes'] = [line[9:].strip() for line in self.banner.proc_card
                                 if line.startswith('generate')]
                data['processes'] += [' '.join(line.split()[2:]) 
                                      for line in self.banner.proc_card
                                      if re.search(r'^\s*add\s+process', line)]
            #object_collector
            #self.id_to_path_second = {}   
            #data['id2path'] = self.id_to_path_second 
        
        # 0. clean previous run ------------------------------------------------
        if not self.rwgt_dir:
            path_me = self.me_dir
        else:
            path_me = self.rwgt_dir
        data['path'] = path_me

        for i in range(2):
            pdir = pjoin(path_me,data['paths'][i])
            if os.path.exists(pdir):
                try:
                    shutil.rmtree(pdir)
                except Exception as error:
                    misc.sprint('fail to rm rwgt dir:', error) 
                    pass 

        # 1. prepare the interface----------------------------------------------
        mgcmd = self.mg5cmd
        complex_mass = False   
        has_cms = re.compile(r'''set\s+complex_mass_scheme\s*(True|T|1|true|$|;)''')
        for line in self.banner.proc_card:
            if line.startswith('set'):
                mgcmd.exec_cmd(line, printcmd=False, precmd=False, postcmd=False)
                if has_cms.search(line):
                    complex_mass = True
            elif line.startswith('define'):
                try:
                    mgcmd.exec_cmd(line, printcmd=False, precmd=False, postcmd=False)
                except madgraph.InvalidCmd:
                    pass 
                          
        # 1. Load model---------------------------------------------------------  
        if  not data['model_name'] and not second:
            raise self.InvalidCmd('Only UFO model can be loaded in this module.')
        elif data['model_name']:
            self.load_model(data['model_name'], data['mg_names'], complex_mass)
            modelpath = self.model.get('modelpath')
            if os.path.basename(modelpath) != mgcmd._curr_model['name']:
                name, restrict = mgcmd._curr_model['name'].rsplit('-',1)
                if os.path.exists(pjoin(os.path.dirname(modelpath),name, 'restrict_%s.dat' % restrict)):
                    modelpath = pjoin(os.path.dirname(modelpath), mgcmd._curr_model['name'])
                
            commandline="import model %s " % modelpath
            if not data['mg_names']:
                commandline += ' -modelname '
            mgcmd.exec_cmd(commandline)
            
            #multiparticles
            for name, content in self.banner.get('proc_card', 'multiparticles'):
                try:
                    mgcmd.exec_cmd("define %s = %s" % (name, content))
                except madgraph.InvalidCmd:
                    pass
                    
        if  second and 'tree_path' in self.dedicated_path:
            files.ln(self.dedicated_path['tree_path'], path_me,name=data['paths'][0])
            if 'virtual_path' in self.dedicated_path:
                has_nlo=True
            else:
                has_nlo=False
        else:
            has_nlo = self.create_standalone_tree_directory(data, second)

        if has_nlo and not self.rwgt_mode:
            self.rwgt_mode = ['NLO']

        # 5. create the virtual for NLO reweighting  ---------------------------
        if second and 'virtual_path' in self.dedicated_path:
            files.ln(self.dedicated_path['virtual_path'], path_me, name=data['paths'][1])
        elif has_nlo and 'NLO' in self.rwgt_mode:
            self.create_standalone_virt_directory(data, second)
            
            if self.multicore == 'create':
                print("compile OLP", data['paths'][1])
                try:
                    misc.compile(['OLP_static'], cwd=pjoin(path_me, data['paths'][1],'SubProcesses'),
                             nb_core=self.mother.options['nb_core'])
                except:
                    misc.compile(['OLP_static'], cwd=pjoin(path_me, data['paths'][1],'SubProcesses'),
                             nb_core=1)
        elif has_nlo and not second and self.rwgt_mode == ['NLO_tree']:
            # We do not have any virtual reweighting to do but we still have to
            #combine the weights.
            #Idea:create a fake directory.
            start = time.time()
            commandline='import model loop_sm;generate g g > e+ ve [virt=QCD]'
            # deactivate golem since it creates troubles
            old_options = dict(mgcmd.options)
            mgcmd.options['golem'] = None             
            commandline = commandline.replace('add process', 'generate',1)
            logger.info(commandline)
            mgcmd.exec_cmd(commandline, precmd=True)
            commandline = 'output standalone_rw %s --prefix=int -f' % pjoin(path_me, data['paths'][1])
            mgcmd.exec_cmd(commandline, precmd=True)    
            #put back golem to original value
            mgcmd.options['golem'] = old_options['golem']
            # update make_opts
            if not mgcmd.options['lhapdf']:
                raise Exception("NLO_tree reweighting requires LHAPDF to work correctly")
            
            # Download LHAPDF SET
            common_run_interface.CommonRunCmd.install_lhapdf_pdfset_static(\
                mgcmd.options['lhapdf'], None, self.banner.run_card.get_lhapdf_id())
            
                
             
        # 6. If we need a new model/process-------------------------------------
        if (self.second_model or self.second_process or self.dedicated_path) and not second :
            self.create_standalone_directory(second=True)    

        if not second:
            self.has_nlo = has_nlo
            
        
        
    def compile(self):
        """compile the code"""
        
        if self.multicore=='wait':
            return
        
        if not self.rwgt_dir:
            path_me = self.me_dir
        else:
            path_me = self.rwgt_dir
        
        rwgt_dir_possibility =   ['rw_me','rw_me_%s' % self.nb_library,'rw_mevirt','rw_mevirt_%s' % self.nb_library]
        for onedir in rwgt_dir_possibility:
            if not os.path.isdir(pjoin(path_me,onedir)):
                continue
            pdir = pjoin(path_me, onedir, 'SubProcesses')
            if self.mother:
                nb_core = self.mother.options['nb_core'] if self.mother.options['run_mode'] !=0 else 1
            else:
                nb_core = 1
            os.environ['MENUM'] = '2'
            misc.compile(['allmatrix2py.so'], cwd=pdir, nb_core=nb_core)
            if not (self.second_model or self.second_process or self.dedicated_path):
                os.environ['MENUM'] = '3'
                misc.compile(['allmatrix3py.so'], cwd=pdir, nb_core=nb_core)

    def load_module(self, metag=1):
        """load the various module and load the associate information"""
        
        if not self.rwgt_dir:
            path_me = self.me_dir
        else:
            path_me = self.rwgt_dir        
        self.id_to_path = {}
        self.id_to_path_second = {}
        rwgt_dir_possibility =   ['rw_me','rw_me_%s' % self.nb_library,'rw_mevirt','rw_mevirt_%s' % self.nb_library]
        for onedir in rwgt_dir_possibility:
            if not os.path.exists(pjoin(path_me,onedir)):
                continue 
            pdir = pjoin(path_me, onedir, 'SubProcesses')
            for tag in [2*metag,2*metag+1]:
                with misc.TMP_variable(sys, 'path', [pjoin(path_me), pjoin(path_me,'onedir', 'SubProcesses')]+sys.path):      
                    mod_name = '%s.SubProcesses.allmatrix%spy' % (onedir, tag)
                    #mymod = __import__('%s.SubProcesses.allmatrix%spy' % (onedir, tag), globals(), locals(), [],-1)
                    if mod_name in list(sys.modules.keys()):
                        del sys.modules[mod_name]
                        tmp_mod_name = mod_name
                        while '.' in tmp_mod_name:
                            tmp_mod_name = tmp_mod_name.rsplit('.',1)[0]
                            del sys.modules[tmp_mod_name]
                        if six.PY3:
                            import importlib
                            mymod = importlib.import_module(mod_name,)
                            mymod = importlib.reload(mymod)
                            #mymod = __import__(mod_name, globals(), locals(), [])
                        else:
                            mymod = __import__(mod_name, globals(), locals(), [],-1) 
                            S = mymod.SubProcesses
                            mymod = getattr(S, 'allmatrix%spy' % tag)
                            reload(mymod) 
                    else:
                        if six.PY3:
                            import importlib
                            mymod = importlib.import_module(mod_name,)
                            #mymod = __import__(mod_name, globals(), locals(), [])    
                        else:
                            mymod = __import__(mod_name, globals(), locals(), [],-1)
                            S = mymod.SubProcesses
                            mymod = getattr(S, 'allmatrix%spy' % tag) 
                    
                
                # Param card not available -> no initialisation
                self.f2pylib[(onedir,tag)] = mymod
                if hasattr(mymod, 'set_madloop_path'):
                    mymod.set_madloop_path(pjoin(path_me,onedir,'SubProcesses','MadLoop5_resources'))
                if (self.second_model or self.second_process or self.dedicated_path):
                    break
            data = self.id_to_path
            if onedir not in ["rw_me",  "rw_mevirt"]:
                data = self.id_to_path_second

            # get all the information
            allids, all_pids = mymod.get_pdg_order()
            all_pdgs = [[pdg for pdg in pdgs if pdg!=0] for pdgs in  allids]
            all_prefix = [bytes(j).decode(errors="ignore").strip().lower() for j in mymod.get_prefix()]
            prefix_set = set(all_prefix)

            hel_dict={}
            for prefix in prefix_set:
                if hasattr(mymod,'%sprocess_nhel' % prefix):
                    nhel = getattr(mymod, '%sprocess_nhel' % prefix).nhel    
                    hel_dict[prefix] = {}
                    for i, onehel in enumerate(zip(*nhel)):
                        hel_dict[prefix][tuple(onehel)] = i+1
                elif hasattr(mymod, 'set_madloop_path') and \
                     os.path.exists(pjoin(path_me,onedir,'SubProcesses','MadLoop5_resources', '%sHelConfigs.dat' % prefix.upper())):
                    hel_dict[prefix] = {}
                    for i,line in enumerate(open(pjoin(path_me,onedir,'SubProcesses','MadLoop5_resources', '%sHelConfigs.dat' % prefix.upper()))):
                        onehel = [int(h) for h in line.split()]
                        hel_dict[prefix][tuple(onehel)] = i+1
                else:
                    misc.sprint(pjoin(path_me,onedir,'SubProcesses','MadLoop5_resources', '%sHelConfigs.dat' % prefix.upper() ))
                    misc.sprint(os.path.exists(pjoin(path_me,onedir,'SubProcesses','MadLoop5_resources', '%sHelConfigs.dat' % prefix.upper())))
                    continue

            for i,(pdg,pid) in enumerate(zip(all_pdgs,all_pids)):
                if self.is_decay:
                    incoming = [pdg[0]]
                    outgoing = pdg[1:]
                else:
                    incoming = pdg[0:2]
                    outgoing = pdg[2:]
                order = (list(incoming), list(outgoing))
                incoming.sort()
                if not self.keep_ordering:
                    outgoing.sort()
                tag = (tuple(incoming), tuple(outgoing))
                if 'virt' in onedir:
                    tag = (tag, 'V')
                prefix = all_prefix[i]
                if prefix in hel_dict:
                    hel = hel_dict[prefix]
                else:
                    hel = {}
                if tag in data:
                    oldpdg = data[tag][0][0]+data[tag][0][1]
                    if all_prefix[all_pdgs.index(pdg)] == all_prefix[all_pdgs.index(oldpdg)]:
                        for i in range(len(pdg)):
                            if pdg[i] == oldpdg[i]:
                                continue
                            if not self.model or not hasattr(self.model, 'get_mass'):
                                continue
                            if self.model.get_mass(int(pdg[i])) == self.model.get_mass(int(oldpdg[i])):
                                continue
                            misc.sprint(tag, onedir)
                            misc.sprint(data[tag][:-1])
                            misc.sprint(order, pdir,)
                            raise Exception                                
                    else: 
                        misc.sprint(all_prefix[all_pdgs.index(pdg)])
                        misc.sprint(all_prefix[all_pdgs.index(oldpdg)])
                        misc.sprint(tag, onedir)
                        misc.sprint(data[tag][:-1])
                        misc.sprint(order, pdir,)
                        raise Exception( "two different matrix-element have the same initial/final state. Leading to an ambiguity. If your events are ALWAYS written in the correct-order (look at the numbering in the Feynman Diagram). Then you can add inside your reweight_card the line 'change keep_ordering True'." )

                data[tag] = order, pdir, hel
             
             
    def load_model(self, name, use_mg_default, complex_mass=False):
        """load the model"""
        
        loop = False

        logger.info('detected model: %s. Loading...' % name)
        model_path = name

        # Import model
        base_model = import_ufo.import_model(name, decay=False,
                                               complex_mass_scheme=complex_mass)
    
        if use_mg_default:
            base_model.pass_particles_name_in_mg_default()
        
        self.model = base_model
        self.mg5cmd._curr_model = self.model
        self.mg5cmd.process_model()
        

    def save_to_pickle(self):
        import madgraph.iolibs.save_load_object as save_load_object
        
        to_save = {}
        to_save['id_to_path'] = self.id_to_path
        if hasattr(self, 'id_to_path_second'):
            to_save['id_to_path_second'] = self.id_to_path_second
        else:
            to_save['id_to_path_second'] = {}
        to_save['all_cross_section'] = self.all_cross_section
        to_save['processes'] = self.processes
        to_save['second_process'] = self.second_process
        if self.second_model:
            to_save['second_model'] =True
        else:
            to_save['second_model'] = None
        to_save['rwgt_dir'] = self.rwgt_dir
        to_save['has_nlo'] = self.has_nlo
        to_save['rwgt_mode'] = self.rwgt_mode
        to_save['rwgt_name'] = self.options['rwgt_name']
        to_save['allow_missing_finalstate'] = self.options['allow_missing_finalstate']
        to_save['identical_particle_in_prod_and_decay'] = self.options['identical_particle_in_prod_and_decay']
        to_save['nb_library'] = self.nb_library

        name = pjoin(self.rwgt_dir, 'rw_me', 'rwgt.pkl')
        save_load_object.save_to_file(name, to_save)


    def load_from_pickle(self, keep_name=False):
        import madgraph.iolibs.save_load_object as save_load_object
        
        obj = save_load_object.load_from_file( pjoin(self.rwgt_dir, 'rw_me', 'rwgt.pkl'))
        
        self.has_standalone_dir = True
        if 'rwgt_info' in self.options:
            self.options = {'rwgt_info': self.options['rwgt_info']}
        else: 
            self.options = {}
        self.options.update({'curr_dir': os.path.realpath(os.getcwd()),
                        'rwgt_name': None})
        
        if keep_name:
            self.options['rwgt_name'] = obj['rwgt_name']


        self.options['allow_missing_finalstate'] = obj['allow_missing_finalstate']
        self.options['identical_particle_in_prod_and_decay'] = obj['identical_particle_in_prod_and_decay']
        old_rwgt = obj['rwgt_dir']
           
        # path to fortran executable
        self.id_to_path = {}
        for key , (order, Pdir, hel_dict) in obj['id_to_path'].items():
            new_P = Pdir.replace(old_rwgt, self.rwgt_dir)
            self.id_to_path[key] = [order, new_P, hel_dict]
            
        # path to fortran executable (for second directory)
        self.id_to_path_second = {}
        for key , (order, Pdir, hel_dict) in obj['id_to_path_second'].items():
            new_P = Pdir.replace(old_rwgt, self.rwgt_dir)
            self.id_to_path_second[key] = [order, new_P, hel_dict]            
        
        self.all_cross_section = obj['all_cross_section']            
        self.processes = obj['processes']
        self.second_process = obj['second_process']
        self.second_model = obj['second_model']
        self.has_nlo = obj['has_nlo']
        self.nb_library = obj['nb_library']
        if not self.rwgt_mode:
            self.rwgt_mode = obj['rwgt_mode']
            logger.info("mode set to %s" % self.rwgt_mode)
        if False:#self.has_nlo and 'NLO' in self.rwgt_mode:
            #use python version
            path = pjoin(obj['rwgt_dir'], 'rw_mevirt','Source')
            sys.path.insert(0, path)
            try:
                mymod = __import__('rwgt2py', globals(), locals())
            except ImportError:
                misc.compile(['rwgt2py.so'], cwd=path)
                mymod = __import__('rwgt2py', globals(), locals())
            with misc.stdchannel_redirected(sys.stdout, os.devnull):
                mymod.initialise([self.banner.run_card['lpp1'], 
                              self.banner.run_card['lpp2']],
                             self.banner.run_card.get_lhapdf_id())
            self.combine_wgt = mymod.get_wgt
                    
        
        
        





        
class DensityInterface(ReweightInterface):
    """Basic interface for computing density matrix"""

    def __init__(self, *args, **opts):
        """init the class"""

        logger.info('Using density mode for reweighting')
        
        self.flag_particle_in_density_matrix = False

        self.helicity_direction = [-1, '', []] #pid of the particle chosen as reference for the helicity frame
        self.particle_in_density_matrix = None #pid of the particles selected for the study
        self.momenta_boost = [-1, '', []] #pid of the particles in whose center of mass frame the system will be boosted
        self.allowed_helicities = None #basis of helicities
        self.spins = None 
        self.number_changing_helicities = None
        self.number_combinations = None
        self.new_param_card = False #Needed to not call ask_edit_card_static
        self.production_matrix = 0
        self.axis_referential = None
        self.symmetrise_initial_state = False
        
        ReweightInterface.__init__(self, *args, **opts)
        self.flag_density_matrix = True
        self.has_run = False

        #This block imports the model, because I need it before do_launch() starts
        mgcmd = self.mg5cmd
        complex_mass = False   
        has_cms = re.compile(r'''set\s+complex_mass_scheme\s*(True|T|1|true|$|;)''')
        for line in self.banner.proc_card:
            if line.startswith('set'):
                mgcmd.exec_cmd(line, printcmd=False, precmd=False, postcmd=False)
                if has_cms.search(line):
                    complex_mass = True
        data = {}
        data['model_name'] = self.banner.get('proc_card', 'model')

        info = self.banner.get('proc_card', 'full_model_line')
        if '-modelname' in info:
            data['mg_names'] = False
        else:
            data['mg_names'] = True
        super().load_model(data['model_name'], data['mg_names'], complex_mass)


    def do_change(self, line):
        """Method called to read the reweight card, redirects to the correct do_change_ method"""
        keyword = line.split()[0]

        if hasattr(self, 'do_change_%s' % keyword):
            return getattr(self, 'do_change_%s' % keyword)(line.split()[1:])
        
        return super().do_change(line)
        

    def find_arrays(self, input_text):
        pattern = re.compile(r"\[[^\]]*\]", re.IGNORECASE)
        return pattern.findall(input_text)

    def find_observable(self, input_text): #we accepect input as "lambda p: p.observable()" or just "observable"
        pattern = re.compile(r"lambda p: p\.[A-Za-z0-9]+", re.IGNORECASE)
        output = pattern.findall(input_text)
        if output == []:
            pattern = re.compile(r"[A-Za-z]+", re.IGNORECASE)
            output = pattern.findall(input_text)
        return pattern.findall(input_text)

    def do_change_helicity_direction(self, line):
        """Change the reference particle for the helicity frame, returns a list of pdg-codes.
        The structure accepted is change helicitty_direction [list of pdg-codes] lambda p: p.pt()
        where the lambda function is any parameter of a particle defined in a lhe file"""
        
        pdg_codes = []
        lambda_function = ''
        order_particles = []
        reconstructed_line = ''

        for i in range(len(line)): #we reconstruct the line to use regex on the line
            reconstructed_line += str(line[i]) + ' '

        observable = self.find_observable(reconstructed_line)
        Arrays = self.find_arrays(reconstructed_line)

        for i in range(len(Arrays)):
            Arrays[i]  = [int(y) for y in Arrays[i].strip("[]").split(',') if y.strip()]
        
        pdg_codes = Arrays[0]
        if len(Arrays) > 1:
            order_particles = Arrays[1]
        else:
            order_particles = []
        if len(observable) > 0:
            lambda_function = observable[0]
        else:
            lambda_function = ''

        #check if the number of ordering parameters is the same as the number of particles selected
        if lambda_function == '' and len(order_particles) > 0:
            logger.error("An order option is given when no observable is selected (helicity referential direction option), the order can not be computed. Please ensure to select an observable among the one defined in the class lhe_parser.FourMomentum")
        if len(order_particles) > 0 and len(order_particles) != len(pdg_codes):
            logger.error("The number of ordering parameters is not the same as the number of particles selected for the helicity referential direction. Please ensure you give the same number of parameters")

        # We don't check what values are put in the arrays, if it is not correct, it will return an error later.

        self.helicity_direction = (pdg_codes, lambda_function, order_particles)


    def do_change_boost_choice(self, line):
        """change the momenta reference for the boost, returns a list of pdg-codes"""
        
        pdg_codes = []
        lambda_function = ''
        order_particles = []
        reconstructed_line = ''

        for i in range(len(line)): #we reconstruct the line to use regex on the line
            reconstructed_line += str(line[i]) + ' '

        observable = self.find_observable(reconstructed_line)
        Arrays = self.find_arrays(reconstructed_line)
        for i in range(len(Arrays)):
            Arrays[i]  = [int(y) for y in Arrays[i].strip("[]").split(',') if y.strip()]
        
        pdg_codes = Arrays[0]
        if len(Arrays) > 1:
            order_particles = Arrays[1]
        else:
            order_particles = []
        if len(observable) > 0:
            lambda_function = observable[0]
        else:
            lambda_function = ''

        #check if the number of ordering parameters is the same as the number of particles selected
        if lambda_function == '' and len(order_particles) > 0:
            logger.error("An order option is given when no observable is selected (boost option), the order can not be computed. Please ensure to select an observable among the one defined in the class lhe_parser.FourMomentum")
        if len(order_particles) > 0 and len(order_particles) != len(pdg_codes): #The code should work even if the two length are different but it is more clear like that for the user too
            logger.error("The number of ordering parameters is not the same as the number of particles selected for the boost. Please ensure you give the same number of parameters")

        # We don't check what values are put in the arrays, if it is not correct, it will return an error later.
        
        self.momenta_boost = [pdg_codes, lambda_function, order_particles]



    def do_change_order_helicities(self, line):
        """Change the order of the basis of helicities. It accepts inputs for density matrices full and partial"""

        for i in range(len(line)): 
            line[i] = int(line[i].strip("[],()"))

        #Let the user enter the allowed_helicities in the complex form ie. [+1, +1, +1, -1, -1, +1, -1, -1] for 2 qubits for instance
        if len(line) == self.number_changing_helicities * self.number_combinations:
            self.allowed_helicities = line
        else: #this part deals with input of the form [basis for particle1] [basis for particle2] 
            cutted_line = []
            counter = 0
            for i in range(len(self.spins)):
                cutted_line.append(line[counter: counter + self.spins[i]])
                counter += self.spins[i]
            
            allowed_hel = []
            for i in range(len(cutted_line[0])):
                for j in range(len(cutted_line[1])):
                    allowed_hel.append(cutted_line[0][i])
                    allowed_hel.append(cutted_line[1][j])
                self.allowed_helicities = allowed_hel
    
    def do_change_symmetrise_initial_state(self, line):
        """
        Chooses whether the initial state should be symmetrised according to 2307.09675. For each event the production matrix calculated is
        R = R(theta) + R(theta + pi)
        """
        for i in range(len(line)): 
            line[i] = line[i].strip("[],()") 
        if line[0] == 'True':
            self.symmetrise_initial_state = True
        elif line[0] == 'False':
            self.symmetrise_initial_state = False
        else:
            logger.warning('Option symmetrise_initial_state not understood, set it to False. Please use the syntax: change symmetrise_initial_state True if you want to enable it.')
            self.symmetrise_initial_state = False
        
        misc.sprint(self.symmetrise_initial_state )

    def do_change_axis_referential(self, line):
        """
        Choses a particle in the initial state that is used as referential to define the production angle theta
        It can be useful for non-symetric initial states like u u~.
        It does accept only one pdg-code
        """
        for i in range(len(line)): 
            line[i] = int(line[i].strip("[],()"))
        self.axis_referential = line
        misc.sprint(self.axis_referential)


    def do_change_particle_in_density_matrix(self, line):
        """change the particle in the density matrix, calculates the number of particles changes,
           their spins and the number of combinations"""
        
        pdg_codes = []
        lambda_function = ''
        order_particles = []
        reconstructed_line = ''

        for i in range(len(line)): #we reconstruct the line to use regex on the line
            reconstructed_line += str(line[i]) + ' '

        observable = self.find_observable(reconstructed_line)
        Arrays = self.find_arrays(reconstructed_line)
        for i in range(len(Arrays)):
            Arrays[i]  = [int(y) for y in Arrays[i].strip("[]").split(',') if y.strip()]
        
        pdg_codes = Arrays[0]
        if len(Arrays) > 1:
            order_particles = Arrays[1]
        else:
            order_particles = []
        if len(observable) > 0:
            lambda_function = observable[0]
        else:
            lambda_function = ''

        #check if the number of ordering parameters is the same as the number of particles selected
        if lambda_function == '' and len(order_particles) > 0:
            logger.error("An order option is given when no observable is selected (particle in density matrix option), the order can not be computed. Please ensure to select an observable among the one defined in the class lhe_parser.FourMomentum")
        if len(order_particles) > 0 and len(order_particles) != len(pdg_codes):
            logger.error("The number of ordering parameters is not the same as the number of particles selected for the particles in density matrix. Please ensure you give the same number of parameters")

        # We don't check what values are put in the arrays, if it is not correct, it will return an error later.

        self.particle_in_density_matrix = (pdg_codes, lambda_function, order_particles) 

        self.number_changing_helicities = len(pdg_codes)

        particles = self.model['particles']

        self.spins = [] #list of spins degrees of freedom for each particle studied
        for particle_id in pdg_codes: #list on the pdg-code of the particles that we study
            for n_particles_model in range(len(particles)):
                if particles[n_particles_model]['pdg_code'] == particle_id or particles[n_particles_model]['pdg_code'] == -particle_id:
                    self.spins.append(particles[n_particles_model]['spin'])

        #Calculation of the number of helicity combinations
        n_comb = 1
        for i in range(len(self.spins)):
            n_comb *= self.spins[i]
        self.number_combinations = n_comb

        #if the user didn't use the option or if it has not been read yet, fill it automatically here
        #how to generalise to graviton ?
        if self.allowed_helicities == None: 
            if self.number_combinations == 2:
                self.allowed_helicities = [+1, -1]
            elif self.number_combinations == 3:
                self.allowed_helicities = [+1, 0, -1]
            elif self.number_combinations == 4:
                self.allowed_helicities = [+1, +1, +1, -1, -1, +1, -1, -1]
            elif self.number_combinations == 6 and self.spins[0] == 2:
                self.allowed_helicities = [+1, +1, +1, 0, +1, -1, -1, +1, -1, 0, -1, -1]
            elif self.number_combinations == 6 and self.spins[0] == 3:
                self.allowed_helicities = [+1, +1, +1, -1, 0, +1, 0, -1, -1, +1, -1, -1]
            elif self.number_combinations == 9:
                self.allowed_helicities = [+1, +1, +1, 0, +1, -1, 0, +1, 0, 0, 0, -1, -1, +1, -1, 0, -1, -1]
            else:
                logger.error("Tried to use density mode selecting more than 2 particles or selecting a spin 0 or spin > 1 particle")
        
        self.flag_particle_in_density_matrix = True


    def do_quit(self, line):
        """exit the reweighting module"""
        #misc.sprint(self.has_run)
        if self.has_run:
            return super().do_quit(line)
        
        if self.particle_in_density_matrix == None:
            logger.error("You have not chosen which particle to put in the density matrix, the density matrix computation can not be done. The command to specify the particles to take is 'change particle_in_density_matrix'.")

        logger.info("helicity_direction = \t" + str(self.helicity_direction))
        logger.info("particle_in_density_matrix = \t" + str(self.particle_in_density_matrix))
        logger.info("momenta_boost = \t" + str(self.momenta_boost))
        logger.info("allowed_helicities = \t" + str(self.allowed_helicities))
        logger.info("spins = \t" + str(self.spins))
        logger.info("number_changing_helicities = \t" + str(self.number_changing_helicities))
        logger.info("number_combinations = \t" + str(self.number_combinations))
        logger.info("axis_referential = \t" + str(self.axis_referential))
        logger.info("symmetrise_initial_state = \t" + str(self.symmetrise_initial_state))

        if self.flag_particle_in_density_matrix == False:
            logger.error("Error: the reweight_card contains no option for the density mode")

        self.has_run = True
        self.run_cmd('launch --keep_card') #calls the function do_launch()


