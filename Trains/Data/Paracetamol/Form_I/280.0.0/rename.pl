for (my $i = 1; $i<=100; $i++) {
	my $name1 = 'dump.'.($i*1000).'.xyz';
	my $name2 = '1.280.0.'.($i*1000).'.Paracetamol.xyz';
	print "mv $name1 $name2\n";
	system("mv $name1 $name2");
}
