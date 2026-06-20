import numpy as np          
import copy                 
import random               

class Genome:

    # This function takes the gene specification from the creature class as input, 
    # along with the evolve mode (only motors/all genes). It returns the indices of 
    # genes to be mutated based on the evolve mode
    @staticmethod
    def get_gene_indices_to_be_mutated(spec, evolve_only_motors):

        # if only motors genes are to be mutated
        if evolve_only_motors == True:
            # returning indices of the three motor related genes
            return [spec[gene]["ind"] for gene in ["control-waveform", "control-amp", "control-freq"]]
        # if all genes are to be mutated
        else:
            # returning all gene indices from the gene specification
            return [spec[gene]["ind"] for gene in spec.keys()]

    @staticmethod
    def get_random_gene(length):
        gene = np.array([np.random.random() for i in range(length)], dtype=float)
        return gene
    
    @staticmethod
    def get_random_genome(gene_length, gene_count):
        genome = [Genome.get_random_gene(gene_length) for i in range(gene_count)]
        return genome

    @staticmethod
    def get_gene_spec():
        gene_spec = {
            "link-shape": {"scale": 1.0},          
            "link-length": {"scale": 2.0},         
            "link-radius": {"scale": 0.5}, # link radius scale reduced to prevent bulky creatures      
            "link-recurrence": {"scale": 1.5}, # link recurrence scale reduced   
            "link-mass": {"scale": 1.0},    
            "joint-type": {"scale": 1.0},          
            "joint-parent": {"scale": 1.0},        
            "joint-axis-xyz": {"scale": 1.0},    
            "joint-origin-rpy-1": {"scale": np.pi * 1.0}, # joint rpy scales reduced 
            "joint-origin-rpy-2": {"scale": np.pi * 1.0}, # to prevent links from
            "joint-origin-rpy-3": {"scale": np.pi * 1.0}, # being too close/colliding  
            "joint-origin-xyz-1": {"scale": 1.0},  
            "joint-origin-xyz-2": {"scale": 1.0},  
            "joint-origin-xyz-3": {"scale": 1.0},  
            "control-waveform": {"scale": 1.0},    
            "control-amp": {"scale": 0.95}, # motor wave amplitude scale increased for greater force      
            "control-freq": {"scale": 13}, # motor wave frequency scale increased for rapid movement
        }

        ind = 0
        for key in gene_spec.keys():
            gene_spec[key]["ind"] = ind  
            ind = ind + 1                    
        return gene_spec  
                   
    @staticmethod
    def get_gene_dict(gene, spec):
        gdict = {}  
        for key in spec:
            ind = spec[key]["ind"]       
            scale = spec[key]["scale"]   
            scaled_value = float(gene[ind]) * float(scale)

            # clipping link length, radius, and mass to ensure
            # they remain stable and do not get too light/small or big/bulky
            if key == "link-length":
                scaled_value = float(np.clip(scaled_value, 0.20, 1.50))       
            if key == "link-radius":
                scaled_value = float(np.clip(scaled_value, 0.04, 0.40))      
            if key == "link-mass":
                scaled_value = float(np.clip(scaled_value, 0.20, 1.50))  

            # ensuring motor waveform amplitude and frequency are 
            # not too small/big, so motors movement is stable
            if key == "control-amp":
                scaled_value = float(np.clip(scaled_value, 0.10, 1.00))       
            if key == "control-freq":
                scaled_value = float(np.clip(scaled_value, 0.50, 15.0))

            # Storing the scaled and clamped gene value
            gdict[key] = scaled_value

        return gdict

    @staticmethod
    def get_genome_dicts(genome, spec):
        gdicts = []
        for gene in genome:
            gdicts.append(Genome.get_gene_dict(gene, spec))
        return gdicts

    @staticmethod
    def expandLinks(parent_link, uniq_parent_name, flat_links, exp_links):

        children = [l for l in flat_links if l.parent_name == parent_link.name]
        sibling_ind = 1

        for c in children:
            for r in range(int(c.recur)):
                sibling_ind  = sibling_ind +1
                c_copy = copy.copy(c)
                c_copy.parent_name = uniq_parent_name
                uniq_name = c_copy.name + str(len(exp_links))
                c_copy.name = uniq_name
                c_copy.sibling_ind = sibling_ind
                exp_links.append(c_copy)
                assert c.parent_name != c.name, (
                    "Genome::expandLinks: link joined to itself: "
                    + c.name + " joins " + c.parent_name
                )
                Genome.expandLinks(c, uniq_name, flat_links, exp_links)

    @staticmethod
    def genome_to_links(gdicts):
        links = []             
        link_ind = 0           
        parent_names = [str(link_ind)]   
        for gdict in gdicts:
            link_name = str(link_ind)  
            parent_ind = gdict["joint-parent"] * len(parent_names)
            assert parent_ind < len(parent_names), "genome.py: parent ind too high: " + str(parent_ind) + "got: " + str(parent_names)
            parent_name = parent_names[int(parent_ind)]
            recur = float(gdict["link-recurrence"])
            link = URDFLink(
                name=link_name,
                parent_name=parent_name,
                recur=recur + 1.0, 
                link_length=gdict["link-length"],
                link_radius=gdict["link-radius"],
                link_mass=gdict["link-mass"],
                joint_type=gdict["joint-type"],
                joint_parent=gdict["joint-parent"],
                joint_axis_xyz=gdict["joint-axis-xyz"],
                joint_origin_rpy_1=gdict["joint-origin-rpy-1"],
                joint_origin_rpy_2=gdict["joint-origin-rpy-2"],
                joint_origin_rpy_3=gdict["joint-origin-rpy-3"],
                joint_origin_xyz_1=gdict["joint-origin-xyz-1"],
                joint_origin_xyz_2=gdict["joint-origin-xyz-2"],
                joint_origin_xyz_3=gdict["joint-origin-xyz-3"],
                control_waveform=gdict["control-waveform"],
                control_amp=gdict["control-amp"],
                control_freq=gdict["control-freq"],
            )

            links.append(link)

            if link_ind != 0:
                parent_names.append(link_name)

            link_ind = link_ind + 1

        if links:
            links[0].parent_name = "None"

        return links

    @staticmethod
    def crossover(g1, g2):
        x1 = random.randint(0, len(g1)-1)
        x2 = random.randint(0, len(g2)-1)
        g3 = np.concatenate((g1[x1:], g2[x2:])) 
        if len(g3) > len(g1):
            g3 = g3[0:len(g1)] 
        return g3
    
    @staticmethod
    def point_mutate(genome, rate, amount, 

                            # taking input of the indices of genes to be mutated  
                            mutation_gene_indices):
        
        mutation_gene_indices = set(int(i) for i in mutation_gene_indices)

        new_genome = copy.copy(genome)
        for gene in new_genome:
            for i in range(len(gene)):

                # skipping genes which are not to be mutated
                if i not in mutation_gene_indices:
                    continue

                if random.random() < rate:
                    gene[i] += random.uniform(-amount, amount)
                if gene[i] >= 1.0:
                    gene[i] = 0.9999
                if gene[i] < 0.0:
                    gene[i] = 0.0
        return new_genome

    @staticmethod
    def shrink_mutate(genome, rate):
        if len(genome) == 1:
            return copy.copy(genome)
        if random.random() < rate:
            ind = random.randint(0, len(genome)-1)
            new_genome = np.delete(genome, ind, 0)
            return new_genome
        else:
            return copy.copy(genome)

    @staticmethod
    def grow_mutate(genome, rate):
        if random.random() < rate:
            gene = Genome.get_random_gene(len(genome[0]))
            new_genome = copy.copy(genome)
            new_genome = np.append(new_genome, [gene], axis=0)
            return new_genome
        else:
            return copy.copy(genome)

    @staticmethod
    def to_csv(dna, csv_file):
        csv_str = ""
        for gene in dna:
            for val in gene:
                csv_str = csv_str + str(val) + ","
            csv_str = csv_str + '\n'

        with open(csv_file, 'w') as f:
            f.write(csv_str)

    @staticmethod
    def from_csv(filename):
        csv_str = ''
        with open(filename) as f:
            csv_str = f.read()   
        dna = []
        lines = csv_str.split('\n')
        for line in lines:
            vals = line.split(',')
            gene = [float(v) for v in vals if v != '']
            if len(gene) > 0:
                dna.append(gene)
        return dna

class URDFLink:
    def __init__(self, name, parent_name, recur,
                 link_length=0.1, 
                 link_radius=0.1, 
                 link_mass=0.5,
                 joint_type=0.1, 
                 joint_parent=0.1, 
                 joint_axis_xyz=0.1,
                 joint_origin_rpy_1=0.1, 
                 joint_origin_rpy_2=0.1, 
                 joint_origin_rpy_3=0.1,
                 joint_origin_xyz_1=0.1, 
                 joint_origin_xyz_2=0.1, 
                 joint_origin_xyz_3=0.1,
                 control_waveform=0.1, 
                 control_amp=0.1, 
                 control_freq=0.1):
        self.name = name
        self.parent_name = parent_name
        self.recur = recur
        self.link_length = float(link_length)
        self.link_radius = float(link_radius)
        self.link_mass = float(link_mass)
        self.joint_type = joint_type
        self.joint_parent = joint_parent
        self.joint_axis_xyz = joint_axis_xyz
        self.joint_origin_rpy_1 = joint_origin_rpy_1
        self.joint_origin_rpy_2 = joint_origin_rpy_2
        self.joint_origin_rpy_3 = joint_origin_rpy_3
        self.joint_origin_xyz_1 = joint_origin_xyz_1
        self.joint_origin_xyz_2 = joint_origin_xyz_2
        self.joint_origin_xyz_3 = joint_origin_xyz_3
        self.control_waveform = control_waveform
        self.control_amp = control_amp
        self.control_freq = control_freq
        self.sibling_ind = 1

    def to_link_element(self, adom):
        link_tag = adom.createElement("link")
        link_tag.setAttribute("name", self.name)
        vis_tag = adom.createElement("visual")
        geom_tag = adom.createElement("geometry")
        cyl_tag = adom.createElement("cylinder")
        cyl_tag.setAttribute("length", str(self.link_length))
        cyl_tag.setAttribute("radius", str(self.link_radius))
        geom_tag.appendChild(cyl_tag)
        vis_tag.appendChild(geom_tag)
        coll_tag = adom.createElement("collision")
        c_geom_tag = adom.createElement("geometry")
        c_cyl_tag = adom.createElement("cylinder")
        c_cyl_tag.setAttribute("length", str(self.link_length))
        c_cyl_tag.setAttribute("radius", str(self.link_radius))
        c_geom_tag.appendChild(c_cyl_tag)
        coll_tag.appendChild(c_geom_tag)
        inertial_tag = adom.createElement("inertial")
        mass_tag = adom.createElement("mass")
        mass_tag.setAttribute("value", str(self.link_mass))

        # Using physics formula to calculate moment of inertia
        # along the longitudinal axis z, based on cylinder radius and mass
        inertia_moment_z = 0.5 * float(self.link_mass) * float(self.link_radius) * float(self.link_radius)

        # Using physics formula to calculate moment of inertia along x/y axis,
        # based on cylinder radius, mass and length
        inertia_moment_x = (1.0 / 12.0) * float(self.link_mass) * (3.0 * float(self.link_radius) * float(self.link_radius) + float(self.link_length) * float(self.link_length))

        inertia_tag = adom.createElement("inertia")
        inertia_tag.setAttribute("ixx", str(inertia_moment_x))
        inertia_tag.setAttribute("iyy", str(inertia_moment_x))
        inertia_tag.setAttribute("izz", str(inertia_moment_z))
        inertia_tag.setAttribute("ixy", "0")
        inertia_tag.setAttribute("ixz", "0")
        inertia_tag.setAttribute("iyx", "0")

        inertial_tag.appendChild(mass_tag)
        inertial_tag.appendChild(inertia_tag)
        link_tag.appendChild(vis_tag)
        link_tag.appendChild(coll_tag)
        link_tag.appendChild(inertial_tag)

        return link_tag

    def to_joint_element(self, adom):

        joint_tag = adom.createElement("joint")
        joint_tag.setAttribute("name", self.name + "_to_" + self.parent_name)
        joint_tag.setAttribute("type", "revolute")
        parent_tag = adom.createElement("parent")
        parent_tag.setAttribute("link", self.parent_name)
        child_tag = adom.createElement("child")
        child_tag.setAttribute("link", self.name)
        axis_tag = adom.createElement("axis")
        if self.joint_axis_xyz <= 0.33:
            axis_tag.setAttribute("xyz", "1 0 0")
        elif self.joint_axis_xyz <= 0.66:
            axis_tag.setAttribute("xyz", "0 1 0")
        else:
            axis_tag.setAttribute("xyz", "0 0 1")

        limit_tag = adom.createElement("limit")
        limit_tag.setAttribute("effort", "1000")
        limit_tag.setAttribute("lower", "-3.1415")
        limit_tag.setAttribute("upper", "3.1415")
        limit_tag.setAttribute("velocity", "200")

        orig_tag = adom.createElement("origin")
        rpy1 = float(self.joint_origin_rpy_1) * float(self.sibling_ind)
        rpy = f"{rpy1} {self.joint_origin_rpy_2} {self.joint_origin_rpy_3}"
        orig_tag.setAttribute("rpy", rpy)

        # Subtracting 0.5 from joint offsets so they may be negative/positive
        # and joints may appear on any side.
        x = float(self.joint_origin_xyz_1) - 0.50 
        
        # Adding further x offset of half of the link length, 
        # to ensure the joint does not appear within the creature
        x = x + 0.50 * float(self.link_length)

        y = float(self.joint_origin_xyz_2) - 0.50 
        z = float(self.joint_origin_xyz_3) - 0.50 

        orig_tag.setAttribute("xyz", f"{x} {y} {z}")

        # attaching created tags to joint
        joint_tag.appendChild(parent_tag)
        joint_tag.appendChild(child_tag)
        joint_tag.appendChild(axis_tag)
        joint_tag.appendChild(limit_tag)
        joint_tag.appendChild(orig_tag)

        return joint_tag
